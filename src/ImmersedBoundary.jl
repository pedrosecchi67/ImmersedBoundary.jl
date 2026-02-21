module ImmersedBoundary

    using ThreadTools

    include("mesher.jl")
    using .Mesher
    export Stereolitography, refine_to_length, merge_points,
        Box, Ball, Line, DistanceField,
        feature_regions, Mesh

    using .Mesher.DocStringExtensions
    using .Mesher.LinearAlgebra

    include("nninterp.jl")
    using .NNInterpolator

    """
    $TYPEDFIELDS

    Struct to define a boundary
    """
    struct Boundary
        points::AbstractMatrix{Float64}
        normals::AbstractMatrix{Float64}
        ghost_indices::AbstractVector{Int64}
        ghost_distances::AbstractVector{Float64}
        image_distances::AbstractVector{Float64}
        image_interpolator::NNInterpolator.Accumulator
    end

    """
    $TYPEDSIGNATURES

    Constructor for a boundary
    """
    function Boundary(
        centers::AbstractMatrix{Float64},
        projections::AbstractMatrix{Float64},
        in_domain::AbstractVector{Bool},
        circumradii::AbstractVector{Float64},
        tree = nothing;
        ghost_layer_ratio::Real = 1.5,
    )
        if isnothing(tree)
            @warn "Building tree in boundary constructor. This is stupid and you shouldn't be doing it."
            tree = KDTree(centers')
        end

        normals = (centers .- projections)
        distances = sum(
            normals .^ 2; dims = 2
        ) |> vec |> x -> sqrt.(x)

        ϵ = eps(Float64)
        normals ./= (distances .+ ϵ)

        # adjust signs as per distance function
        @. distances *= 2 * (in_domain - 0.5)
        @. normals *= 2 * (in_domain - 0.5)

        image_distances = @. ghost_layer_ratio * circumradii * 2

        ghost_indices = (
            @. image_distances >= abs(distances)
        ) |> findall

        normals = normals[ghost_indices, :]
        distances = distances[ghost_indices]
        image_distances = image_distances[ghost_indices]
        projections = centers[ghost_indices, :] .- normals .* distances

        Boundary(
            projections,
            normals, 
            ghost_indices,
            distances,
            image_distances,
            Interpolator(
                centers, projections .+ image_distances .* normals, tree;
                linear = true, first_index = true
            ),
        )
    end

    export cruciform_stencil

    """
    $TYPEDFIELDS

    Function to instantiate a cruciform stencil,
    identified by tuples of relative point positions in 
    a finite-difference grid
    """
    cruciform_stencil(
        ndim::Int, order::Int = 2
    ) = let T = NTuple{ndim, Int64}
        stencil = T[]

        for dim = 1:ndim
            for i = (-order:order)
                pt = zeros(Int64, ndim)
                pt[dim] = i

                push!(
                    stencil,
                    tuple(pt...)
                )
            end
        end

        unique(stencil)
    end

    """
    $TYPEDFIELDS

    Struct to define a partition
    """
    struct Partition
        index::Int64
        image::AbstractVector{Int64}
        image_in_domain::AbstractVector{Int64}
        skirt_indices::AbstractVector{Int64}
        domain::AbstractVector{Int64}
        stencils::AbstractDict
        centers::AbstractMatrix{Float64}
        spacing::AbstractMatrix{Float64}
        boundaries::Dict{String, Boundary}
    end

    """
    $TYPEDSIGNATURES

    Constructor for a partition.

    The stencil should be a vector of tuples with relative
    point indexing for finite difference stencils. If not provided,
    a second-order, cruciform stencil will be used (see `cruciform_stencil()`).
    """
    function Partition(
        msh::Mesh, indices;
        index::Int64 = 0,
        stencil = nothing,
        tree = nothing,
        ghost_layer_ratio::Real = 1.5
    )
        nd = size(msh.centers, 1)

        if isnothing(stencil)
            stencil = cruciform_stencil(nd)
        end

        image = collect(indices)

        stencil_interpolators = Dict{
            eltype(stencil), NNInterpolator.Accumulator
        }()

        if isnothing(tree)
            @warn "Building KD-tree once per partition. This is inefficient"

            tree = KDTree(msh.centers)
        end

        domain, image_in_domain, this_tree = let points = permutedims(msh.centers)
            widths = permutedims(msh.widths)

            # build first batch of interpolators to detect domain
            this_points = @view points[image, :]
            this_widths = @view widths[image, :]

            interpolators = NNInterpolator.Accumulator[]
            for stpoint in stencil
                v = collect(stpoint)

                intp = Interpolator(
                    points, this_points .+ this_widths .* v', tree;
                    first_index = true, linear = false,
                )
                push!(interpolators, intp)
            end

            domain, hmap = NNInterpolator.domain(interpolators...)

            # now, re-build interpolators considering domain
            this_points = @view points[domain, :]
            this_widths = @view widths[domain, :]

            this_tree = KDTree(this_points')

            for stpoint in stencil
                v = collect(stpoint)

                intp = Interpolator(
                    this_points, this_points .+ this_widths .* v', this_tree;
                    first_index = true, linear = false,
                )

                stencil_interpolators[stpoint] = intp
            end

            image_in_domain = map(i -> hmap[i], image)

            (domain, image_in_domain, this_tree)
        end

        skirt_indices = setdiff(1:length(domain), image_in_domain)

        msh = msh[domain]

        centers = msh.centers |> permutedims
        spacing = msh.widths |> permutedims
        circumradii = sum(
            spacing .^ 2; dims = 2
        ) |> vec |> x -> sqrt.(x) ./ 2

        boundaries = Dict{String, Boundary}()
        for (bname, bprojs) in msh.family_projections
            boundaries[bname] = Boundary(
                centers, permutedims(bprojs), msh.in_domain, circumradii,
                this_tree;
                ghost_layer_ratio = ghost_layer_ratio,
            )
        end

        Partition(
            index,
            image, image_in_domain, skirt_indices,
            domain,
            stencil_interpolators,
            centers, spacing,
            boundaries,
        )
    end

    """
    $TYPEDSIGNATURES

    Obtain values at a given stencil point within a partition.

    Example:

    ```
    u = rand(length(domain))
    ux = similar(u)

    domain(u, ux) do part, u, ux # values at local domain
        dx, dy = part.spacing |> eachcol

        ux .= ( # note that we're editing in-place
            part(u, 1, 0) .- part(u, -1, 0)
        ) ./ (2 .* dx)
    end

    # ux is now the first, x-axis derivative of u
    ```
    """
    function (part::Partition)(
        U::AbstractArray, inds::Int...
    )
        nd = size(part.spacing, 2)
        pt = zeros(Int32, nd)
        for (k, i) in enumerate(inds)
            pt[k] = i
        end
        pt = tuple(pt...)

        part.stencils[pt](U)
    end

    export getalong

    """
    $TYPEDSIGNATURES

    Obtain stencil index `i` along dimension `dim` in a partition array.
    `part(u, 0, 3, 0)` is equivalent to `ibm.getalong(part, u, 2, 3)`, for example
    """
    getalong(part::Partition, U::AbstractArray, dim::Int, i::Int) = let inds = zeros(
        Int64, dim
    )
        inds[end] = i
        part(U, inds...)
    end

    export impose_bc!

    """
    $TYPEDSIGNATURES

    Impose boundary condition on domain array.

    Example for non-penetration condition:

    ```
    dom(u, v) do part, udom, vdom
        # function receives values of field properties at image points
        # and returns their values at the boundary
        ibm.impose_bc!(part, "wall", udom, vdom) do bdry, uimage, vimage
            nx, ny = bdry.normals |> eachcol
            un = @. nx * uimage + ny * vimage

            (
                uimage .- un .* nx,
                vimage .- un .* ny
            )
        end
    end

    # alternative return value:
    uv = zeros(length(dom), 2)
    uv[:, 1] .= 1.0
    dom(uv) do part, uvdom
        ibm.impose_bc!(part, "wall", uvdom) do bdry, uvim
            uimage, vimage = eachcol(uvim)
            nx, ny = eachcol(bdry.normals)
            un = @. nx * uimage + ny * vimage

            uvim .- un .* bdry.normals
        end
    end
    ```

    Kwargs are passed directly to the BC function.
    Note that other field variable args. may be passed
    as auxiliary variables (e. g. the BC function may receive
    3 arrays as an input, and return BCs solely for the first two).

    We may directly return ghost cell values rather than boundary values
    by activating flag `impose_at_ghost`.
    """
    function impose_bc!(
        f,
        part::Partition, bname::String,
        args::AbstractArray{Float64}...;
        impose_at_ghost::Bool = false,
        kwargs...
    )
        bdry = part.boundaries[bname]

        iargs = bdry.image_interpolator.(args)
        bargs = f(
            bdry, iargs...; kwargs...
        )

        if bargs isa AbstractArray
            if eltype(bargs) <: Number
                bargs = (bargs,) # single return value! single variable to set
            end
        end

        if impose_at_ghost
            for (a, b) in zip(args, bargs)
                selectdim(
                    a, 1, bdry.ghost_indices
                ) .= b
            end
        else
            η = bdry.ghost_distances ./ bdry.image_distances

            for (a, b, i) in zip(args, bargs, iargs)
                selectdim(
                    a, 1, bdry.ghost_indices
                ) .= i .* η .+ (1.0 .- η) .* b
            end
        end
    end

    """
    $TYPEDFIELDS
    
    Struct to define a surface for postprocessing of volumetric data
    """
    struct Surface
        stl::Stereolitography
        centers::AbstractMatrix{Float64}
        normals::AbstractMatrix{Float64}
        areas::AbstractVector{Float64}
        offsets::AbstractVector{Float64}
        interpolator::NNInterpolator.Accumulator
        offset_interpolator::NNInterpolator.Accumulator
    end

    """
    $TYPEDSIGNATURES

    Constructor for a surface
    """
    function Surface(
        stl::Stereolitography,
        cell_centers::AbstractMatrix{Float64},
        widths::AbstractMatrix{Float64},
        tree = nothing;
        ghost_layer_ratio::Real = 1.5,
    )
        if isnothing(tree)
            @warn "Constructing trees in surface constructor. Stupid! Stupid! Stupid!"
            tree = KDTree(cell_centers')
        end

        circumdiameters = sum(
            widths .^ 2; dims = 2
        ) |> vec |> x -> sqrt.(x)

        centers, normals = centers_and_normals(stl)
        areas = sum(
            normals .^ 2; dims = 1
        ) |> vec |> x -> sqrt.(x)

        ϵ = eps(Float64)
        centers = permutedims(centers)
        normals = permutedims(normals)
        normals ./= (areas .+ ϵ)

        interpolator = Interpolator(
            cell_centers, centers, tree; first_index = true, linear = true
        )

        offsets = ghost_layer_ratio .* interpolator(circumdiameters)
        offset_interpolator = Interpolator(
            cell_centers, centers .+ normals .* offsets, tree; 
            first_index = true, linear = true
        )

        Surface(
            stl, centers, normals, areas,
            offsets, interpolator, offset_interpolator
        )
    end

    export at_offset

    """
    $TYPEDSIGNATURES

    Obtain values of field property an offset away from the surface (see vector 
    `surf.offsets`). The first index should refer to the cell/surface index.
    """
    at_offset(
        surf::Surface, u::AbstractArray
    ) = surf.offset_interpolator(u)

    """
    $TYPEDSIGNATURES

    Interpolate a field property to a surface. The first index should refer to 
    cell/surface point index.
    """
    (surf::Surface)(u::AbstractArray) = surf.interpolator(u)

    export surface_integral

    """
    $TYPEDSIGNATURES

    integrate a property throughout a surface
    """
    surface_integral(surf::Surface, u::AbstractVector) = (surf.areas .* u |> sum)

    """
    $TYPEDSIGNATURES

    integrate a property throughout a surface. The first dimension in the array
    is assumed to refer to point/cell indices
    """
    surface_integral(surf::Surface, u::AbstractArray) = (
        surf.areas .* u |> a -> sum(a; dims = 1) |> a -> dropdims(a; dims = 1)
    )

    export Domain

    """
    $TYPEDFIELDS

    Struct to define a domain
    """
    struct Domain
        mesh::Mesh
        partitions::AbstractDict{Int64, Partition}
        surfaces::Dict{String, Surface}
    end

    """
    Create ranges defining which block goes to which partition
    """
    partition_ranges(N::Int, nmax::Int) = let nparts = (
        N ÷ nmax + 1
    )
        dn = N ÷ nparts
        n0 = 0

        pranges = []
        for i = 1:nparts
            push!(
                pranges,
                (n0 + 1):(
                    i == nparts ?
                    N :
                    (n0 + dn)
                )
            )

            n0 += dn
        end

        pranges
    end

    """
    $TYPEDSIGNATURES

    Constructor for a domain.

    `ghost_layer_ratio` is the ratio between image point distances to the wall and the
    ghost cells' circumradiameters.
    """
    function Domain(
        msh::Mesh;
        stencil = nothing,
        max_partition_size::Int = 1000_000,
        ghost_layer_ratio::Real = 1.5,
    )
        pranges = partition_ranges(length(msh), max_partition_size)

        tree = KDTree(msh.centers)

        partitions = map(
            i -> Partition(
                msh, pranges[i]; stencil = stencil, tree = tree, index = i,
                ghost_layer_ratio = ghost_layer_ratio,
            ), 1:length(pranges)
        )
        partitions = Dict( # to dictionary in order to enable custom backends
            i => p for (i, p) in enumerate(partitions)
        )

        surfaces = Dict{String, Surface}()
        for (fname, stl) in msh.families
            surfaces[fname] = Surface(
                stl, 
                msh.centers |> permutedims,
                msh.widths |> permutedims,
                tree; ghost_layer_ratio = ghost_layer_ratio
            )
        end

        Domain(
            msh, partitions, surfaces
        )
    end

    """
    $TYPEDSIGNATURES

    Get number of cells in domain
    """
    Base.length(dom::Domain) = length(dom.mesh)

    """
    $TYPEDSIGNATURES

    Get dimensionality of domain
    """
    Base.ndims(dom::Domain) = size(dom.mesh.centers, 1)

    """
    $TYPEDSIGNATURES

    Run a loop over the partitions of a domain and
    execute operations.

    Example:

    ```
    domain(r, u) do partition, rdom, udom
        # udom includes the parts of vector `u`
        # which affect the residual at partition `partition`.

        # now do some Cartesian grid operations and
        # update rdom
    end

    # after the loop, the values of `rdom` are returned to
    # array `r`
    ```

    This allows for large operations on field data
    to be performed one partition at a time,
    saving on max. memory usage.

    Return values are also stored in a vector, which is
    then returned.
    Kwargs are passed to the called function.

    Ran with `n_threads` treads.
    """
    (dom::Domain)(
        f, args::AbstractArray...; 
        n_threads::Int = 1,
        kwargs...
    ) = tmap(
        i -> let part = dom.partitions[i]
            vpargs = map(
                a -> selectdim(a, 1, part.domain), args
            )
            pargs = copy.(vpargs)

            r = f(
                part,
                pargs...; kwargs...
            )

            for (v, a) in zip(vpargs, pargs)
                selectdim(
                    v, 1, part.image_in_domain
                ) .= selectdim(
                    a, 1, part.image_in_domain
                )
            end

            r
        end,
        n_threads,
        1:length(dom.partitions)
    )

    export export_vtk

    """
    $TYPEDSIGNATURES

    Export surfaces and volume data to VTK file within a given folder.
    Kwargs are treated as field properties (cell data).

    Surface-specific, externally-processed data may be passed with `surface_data`:

    ```
    surface1 = domain.surfaces["surface1"]
    surface2 = domain.surfaces["surface2"]

    u = rand(length(domain))
    v = rand(length(domain))

    surface_data = Dict(
        "surface1" => (
            u = surface1(u),
            v = surface1(v),
        ),
        "surface2" => (
            u = surface2(u),
            v = surface2(v),
        )
    )
    ```
    """
    function export_vtk(
        folder::String,
        dom::Domain;
        include_volume::Bool = true,
        include_surface::Bool = true,
        surface_data::AbstractDict = Dict{String, AbstractArray}(),
        kwargs...
    )
        if isdir(folder)
            @warn "Overwriting VTK data in folder $folder"
            rm(folder; recursive = true)
        end
        mkdir(folder)

        field_data = Dict(
            [
                k => let pdims = circshift(1:ndims(v), -1) |> x -> tuple(x...)
                    permutedims(v, pdims)
                end for (k, v) in kwargs
            ]...
        )

        if include_volume
            grid = Mesher.vtk_grid(
                joinpath(folder, "volume"), dom.mesh; field_data...
            )
            Mesher.vtk_save(grid)
        end

        if include_surface
            vtm = Mesher.WriteVTK.vtk_multiblock(joinpath(folder, "surface"))

            for (sname, surf) in dom.surfaces
                surf_data = Dict(
                    [
                        k => let pdims = circshift(1:ndims(v), -1) |> x -> tuple(x...)
                            permutedims(
                                (
                                    size(v, 1) > length(surf.offsets) ?
                                    surf(v) : v
                                ), 
                                pdims)
                        end for (k, v) in kwargs
                    ]...
                )

                if haskey(surface_data, sname)
                    fam_data = surface_data[sname]
                    let ks = propertynames(fam_data)
                        for k in ks
                            v = getproperty(fam_data, k)
                            
                            surf_data[k] = let pdims = circshift(1:ndims(v), -1) |> x -> tuple(x...)
                                permutedims(
                                    (
                                        size(v, 1) > length(surf.offsets) ?
                                        surf(v) : v
                                    ), 
                                    pdims)
                            end
                        end
                    end
                end

                grid = Mesher.vtk_grid(
                    joinpath(folder, sname), surf.stl, vtm;
                    surf_data...)
            end

            Mesher.WriteVTK.vtk_save(vtm)
        end
    end

    export ∇, Δ, δ, μ, MUSCL, laplacian_smoothing

    """
    $TYPEDSIGNATURES

    Obtain a backward derivative along dimension `dim`.
    """
    ∇(part::Partition, u::AbstractArray, dim::Int64) = let uim1 = getalong(part, u, dim, -1)
        (part(u) .- uim1) ./ part.spacing[:, dim]
    end

    """
    $TYPEDSIGNATURES

    Obtain a forward derivative along dimension `dim`.
    """
    Δ(part::Partition, u::AbstractArray, dim::Int64) = let uip1 = getalong(part, u, dim, 1)
        (uip1 .- part(u)) ./ part.spacing[:, dim]
    end

    """
    $TYPEDSIGNATURES

    Obtain a central derivative along dimension `dim`.
    """
    δ(part::Partition, u::AbstractArray, dim::Int64) = let uip1 = getalong(part, u, dim, 1)
        (uip1 .- getalong(part, u, dim, -1)) ./ (
            2 .* part.spacing[:, dim]
        )
    end

    """
    $TYPEDSIGNATURES

    Obtain the average of a property at face `i + 1/2` along dimension `dim`.
    """
    μ(part::Partition, u::AbstractArray, dim::Int64) = (
        getalong(part, u, dim, 1) .+ part(u)
    ) ./ 2

    """
    $TYPEDSIGNATURES

    Run iteration of laplacian smoothing (obtain average of neighbors)
    """
    function laplacian_smoothing(part::Partition, u::AbstractArray)
        uavg = similar(u)
        uavg .= 0.0

        cnt = 0
        for i = 1:size(part.spacing, 2)
            cnt += 2
            uavg .+= (
                getalong(part, u, i, -1) .+ getalong(part, u, i, 1)
            )
        end

        uavg ./ cnt
    end

    export stencil_average

    """
    $TYPEDSIGNATURES

    Run averaging over all neighbors in the finite difference stencil. 
    Useful for smoothing in a single iteration.
    """
    function stencil_average(part::Partition, u::AbstractArray)
        uavg = similar(u)
        uavg .= 0.0

        cnt = 0
        for t in part.stencils |> keys
            cnt += 1
            uavg .+= part(u, t...)
        end

        uavg ./ cnt
    end

    """
    Minmod operator
    """
    minmod(u1::Real, u2::Real) = min(abs(u1), abs(u2)) * (sign(u1) + sign(u2)) / 2

    """
    $TYPEDSIGNATURES

    Obtain values at left and right face of a cell
    using MUSCL reconstruction, given its neighbors. Uses minmod
    limiter
    """
    function MUSCL(uim1::AbstractArray, ui::AbstractArray, uip1::AbstractArray)
        grad = @. minmod(ui - uim1, uip1 - ui)

        (
            ui .- grad ./ 2,
            ui .+ grad ./ 2,
        )
    end

    export advection

    """
    $TYPEDSIGNATURES

    Auxiliary function for `-∇⋅(uϕ)`.
    Uses upwinding (`order = 1`) or linear-upwinding with MUSCL (`order = 2`).
    """
    function advection(
        part::Partition, u::AbstractMatrix, ϕ::AbstractArray;
        order::Int = 2,
    )
        mdiv = similar(ϕ)
        mdiv .= 0.0

        for dim = 1:size(u, 2)
            v = @view u[:, dim]
            h = @view part.spacing[:, dim]

            vim12 = (getalong(part, v, dim, -1) .+ v) ./ 2
            vip12 = (getalong(part, v, dim, 1) .+ v) ./ 2

            ϕLim12 = ϕRim12 = ϕLip12 = ϕRip12 = nothing
            if order == 1
                ϕLim12 = getalong(part, ϕ, dim, -1)
                ϕRim12 = ϕ
                ϕLip12 = ϕ
                ϕRip12 = getalong(part, ϕ, dim, 1)
            elseif order == 2
                ϕim2 = getalong(part, ϕ, dim, -2)
                ϕim1 = getalong(part, ϕ, dim, -1)
                ϕip1 = getalong(part, ϕ, dim, 1)
                ϕip2 = getalong(part, ϕ, dim, 2)

                _, ϕLim12 = MUSCL(ϕim2, ϕim1, ϕ)
                ϕRim12, ϕLip12 = MUSCL(ϕim1, ϕ, ϕip1)
                ϕRip12, _ = MUSCL(ϕ, ϕip1, ϕip2)
            else
                throw(error("Order $order unsupported for advection-dissipation"))
            end
            
            @. mdiv -= (
                (
                    vip12 * (ϕLip12 + ϕRip12) - abs(vip12) * (ϕRip12 - ϕLip12)
                ) - (
                    vim12 * (ϕLim12 + ϕRim12) - abs(vim12) * (ϕRim12 - ϕLim12)
                )
            ) / 2 / h
        end

        mdiv
    end

    export dissipation

    """
    $TYPEDSIGNATURES

    Auxiliary function for `-∇⋅(μ ∇ϕ)`.
    """
    function dissipation(
        part::Partition, μ::Union{Real, AbstractMatrix}, ϕ::AbstractArray
    )
        div = similar(ϕ)
        div .= 0.0

        for dim = 1:size(part.spacing, 2)
            h = @view part.spacing[:, dim]

            μim1 = μip1 = μ
            if μ isa AbstractArray
                μim1 = getalong(part, μ, dim, -1)
                μip1 = getalong(part, μ, dim, 1)
            end

            ϕim1 = getalong(part, ϕ, dim, -1)
            ϕip1 = getalong(part, ϕ, dim, 1)

            @. div += (
                (μip1 + μ) * (ϕip1 - ϕ) -
                (μim1 + μ) * (ϕ - ϕim1) 
            ) / h ^ 2 / 2
        end

        div
    end

    include("arraybends.jl")
    using .ArrayBackends

    export to_backend

    @declare_converter NNInterpolator.Accumulator
    @declare_converter Boundary
    @declare_converter Partition
    @declare_converter Surface
    @declare_converter Domain

    include("cfd.jl")
    using .CFD

    include("turbulence.jl")
    using .Turbulence

end
