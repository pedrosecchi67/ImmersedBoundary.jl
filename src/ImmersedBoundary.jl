module ImmersedBoundary

    include("mesher.jl")
    import .Mesher as mshr
    using .Mesher: Stereolitography, Ball, Box, Line, Triangulation,
        PlaneSDF, TriReference, Projection
    using .Mesher.WriteVTK
    using .Mesher.STLHandler: STLTree, point_in_polygon, 
        refine_to_length, stl2vtk, centers_and_normals, feature_edges

    using .Mesher.DocStringExtensions
    using .Mesher.LinearAlgebra
    using .Mesher: @threads

    include("nninterp.jl")
    using .NNInterpolator

    using Base.Iterators
    using Serialization

    """
    $TYPEDFIELDS

    Struct to define a boundary
    """
    struct Boundary
        image_interpolator::Interpolator
        points::AbstractMatrix{Float64}
        normals::AbstractMatrix{Float64}
        image_distances::AbstractVector{Float64}
        ghost_distances::AbstractVector{Float64}
        ghost_indices::AbstractVector{Int64}
    end

    """
    $TYPEDSIGNATURES

    Create a boundary from partition info
    """
    function Boundary(
        circumradius::AbstractVector{Float64},
        X::AbstractMatrix{Float64}, 
        projs::AbstractMatrix{Float64},
        sdfs::AbstractVector{Float64},
        tree::KDTree,
        ghost_ratios::Tuple{Float64, Float64};
        Xc::Union{Nothing, AbstractMatrix{Float64}} = nothing,
    )
        if isnothing(Xc)
            Xc = X
        end

        #check for ghost indices between circumradius thresholds
        g1, g2 = ghost_ratios
        ghost_indices = findall(
            (@. (sdfs <= g2 * circumradius) && (sdfs >= g1 * circumradius))
        )

        # filter arrays to ghosts only
        projs = projs[ghost_indices, :]
        ghosts = X[ghost_indices, :]
        circumradius = circumradius[ghost_indices]

        # obtain normals and ghost distances
        normals = @. (ghosts - projs)
        ghost_distances = sdfs[ghost_indices]

        ϵ = eps(Float64) # regularize
        @. normals /= (ghost_distances + sign(ghost_distances) * ϵ)

        # obtain image distance as per heuristics
        sqnd = sqrt(2.0) # sqrt(size(normals, 2))
        image_distances = @. max(
            circumradius * sqnd, ghost_distances + circumradius * sqnd * 1.1
        )

        image_points = @. projs + normals * image_distances

        # construct image interpolator
        intp = Interpolator(Xc, image_points, tree; first_index = true)

        Boundary(
            intp, ghosts, normals, 
            image_distances, ghost_distances,
            ghost_indices
        )
    end

    """
    $TYPEDFIELDS

    Struct representing a surface for property integration and postprocessing
    """
    struct Surface
        stereolitography::Mesher.Stereolitography
        points::AbstractMatrix{Float64}
        normals::AbstractMatrix{Float64}
        areas::AbstractVector{Float64}
        interpolator::Interpolator
        offsets::AbstractVector{Float64}
        offset_interpolator::Interpolator
    end

    """
    $TYPEDSIGNATURES

    Obtain a surface from a domain and a stereolitography object.

    If `max_length` is provided, the STL surface is refined by tri splitting until no
    triangle side is larger than the provided value.
    """
    function Surface(
        centers::AbstractMatrix{Float64}, spacing::AbstractMatrix{Float64},
        tree::KDTree, 
        stl::Mesher.Stereolitography; max_length::Float64 = 0.0
    )

        if max_length > 0.0
            stl = refine_to_length(stl, max_length)
        end

        nd = size(stl.points, 1)
        points = permutedims(stl.points)

        interpolator = Interpolator(centers, points, tree; first_index = true)

        _, normals = centers_and_normals(stl)

        normals = let point_normals = similar(stl.points)
            point_normals .= 0.0
            
            for ipts in eachrow(stl.simplices)
                for (n, ipt) in zip(eachcol(normals), ipts)
                    point_normals[:, ipt] .+= n ./ nd
                end
            end

            permutedims(point_normals)
        end

        ϵ = eltype(normals) |> eps
        areas = map(
                    n -> norm(n) + ϵ, eachrow(normals)
        )
        normals = normals ./ areas

        circumdiameters = sum(spacing .^ 2; dims = 2) |> vec |> x -> sqrt.(x)
        offsets = interpolator(circumdiameters)
        offset_interpolator = Interpolator(centers, points .+ normals .* offsets,
            tree; first_index = true)

        Surface(
            deepcopy(stl),
            points,
            normals,
            areas,
            interpolator,
            offsets, offset_interpolator
        )

    end

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

    """
    $TYPEDFIELDS

    Type to represent a mesh parition.
    """
    struct Partition{N}
        domain::AbstractVector{Int32}
        image::AbstractVector{Int32}
        image_in_domain::AbstractVector{Int32}
        stencils::Dict{NTuple{N, Int32}, Interpolator}
        boundaries::Dict{String, Boundary}
        spacing::AbstractMatrix{Float64}
        centers::AbstractMatrix{Float64}
    end

    """
    $TYPEDFIELDS

    Struct defining a domain
    """
    struct Domain
        ndofs::Int64
        mesh::Mesher.Mesh
        partitions::AbstractVector{Partition}
        surfaces::Dict{String, Surface}
        boundary_distances::Dict{String, AbstractVector{Float64}}
    end

    """
    $TYPEDSIGNATURES

    Obtain number of DOFs in a domain
    """
    Base.length(dom::Domain) = dom.ndofs

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
        
    Generate an octree mesh defined by:

    * A hypercube origin;
    * A vector of hypercube widths;
    * A set of tuples in format `(name, surface, max_length)` describing
        stereolitography surfaces (`Mesher.Stereolitography`) and 
        the max. cell widths at these surfaces;
    * A set of refinement regions described by distance functions and
        the local refinement at each region. Example:
            ```
            refinement_regions = [
                ibm.Ball([0.0, 0.0], 0.1) => 0.005,
                ibm.Ball([1.0, 0.0], 0.1) => 0.005,
                ibm.Box([-1.0, -1.0], [3.0, 2.0]) => 0.0025,
                ibm.Line([1.0, 0.0], [2.0, 0.0]) => 0.005
            ]
            ```
    * A maximum cell size (optional);
    * A volumetric growth ratio;
    * An interval of ratios between the cell circumradius and the SDF of a given surface for 
        which cells are defined as ghosts;
    * A point reference within the domain. If absent, external flow is assumed;
    * An approximation ratio between wall distance and cell circumradius past which
        distance functions are approximated; 
    * A maximum number of cells per partition;
    * A set of families defining surface groups for postprocessing, BC imposition and wall
        distance calculations; and
    * An optional tuple specifying the number of "splits" conducted along each axis
        before octree splitting. For example, if one has `origin = [1.0, 1.0]`,
        `widths = [2.0, 3.0]`, one may use `initial_splits = (2, 3)` to maintain isotropy.

    The families may be defined with the following syntax:

    ```
    surfaces = [
        ("wing", "wing.stl", 1e-3),
        ("flap", "flap.stl", 1e-3)
    ]
    families = [
        "wall" => ["wing", "flap"], # a group of surface names
        "inlet" => [
            (1, false), # fwd face, first dimension (x)
            (2, false), # left face, second dimension (y)
            (2, true), # right face, second dimension (y)
            (3, false), # bottom face, third dimension (z)
            (3, true), # top face, third dimension (z)
        ],
        "outlet" => [(1, true)]
    ]

    dom = ibm.Domain(origin, widths, surfaces...; families = families)
    ```

    The default family definition uses each surface as a family and defines
    the farfield as `"FARFIELD"`.
    
    Stencil points may be specified as tuples. An example for first order operations
    on a 3D mesh is:

    ```
    stencil_points = [
        (-1, 0, 0), (0, 0, 0), (1, 0, 0),
        (0, -1, 0), (0, 0, 0), (0, 1, 0), # don't worry about duplicate values
        (0, 0, -1), (0, 0, 0), (0, 0, 1)
    ]
    ```

    The default is a cruciform, second-order CFD stencil.
    """
    function Domain(
            origin::Vector{Float64}, widths::Vector{Float64},
            surfaces::Tuple{String, Stereolitography, Float64}...;
            initial_splits = nothing,
            refinement_regions::AbstractVector = [],
            max_length::Float64 = Inf,
            growth_ratio::Float64 = 1.1,
            ghost_layer_ratio::Tuple = (-1.1, 2.1),
            interior_point = nothing,
            approximation_ratio::Float64 = 2.0,
            verbose::Bool = false,
            max_partition_cells::Int64 = 1000_000,
            families = nothing,
            stencil_points = Tuple[],
    )
        nd = length(origin)

        # fill stencil points if unspecified
        if length(stencil_points) == 0
            stencil_points = copy(stencil_points) # let's not mess with the kwarg def. value

            pt = zeros(Int32, nd)
            for i = 1:nd
                for k = -2:2
                    pt[i] = k
                    push!(stencil_points, tuple(pt...))
                    pt[i] = 0
                end
            end
        end
        stencil_points = unique(stencil_points)

        # define surfaces and families
        stl_dict = Dict( # store all STLs in a dictionary. We'll need them later for fam. construction
            [
                bname => stl for (bname, stl, _) in surfaces
            ]...
        )

        if isnothing(families) # default: FARFIELD + whichever surfaces you have
            if haskey(stl_dict, "FARFIELD")
                throw(error("Surface definition uses reserved name FARFIELD"))
            end

            families = Dict{String, AbstractVector}(
                "FARFIELD" => collect(
                    Iterators.product(1:nd, (false, true))
                ) |> vec
            )

            for (bname, _, _) in surfaces
                families[bname] = [bname]
            end
        end

        if !(families isa AbstractDict)
            families = Dict(families...)
        end

        if haskey(stl_dict, "surface") || haskey(families, "surface")
            throw(error("Surface/family name \'surface\' is reserved"))
        end

        # generate octree
        if verbose
            println("Generating octree...")
        end

        # separate hypercube boundaries
        hypercube_boundaries = Dict{String, AbstractVector}()
        for (fam, famdef) in families
            if eltype(famdef) <: Tuple
                hypercube_boundaries[fam] = famdef
            end
        end

        msh = mshr.FixedMesh(
            origin, widths,
            surfaces...;
            initial_splits = initial_splits,
            growth_ratio = growth_ratio,
            refinement_regions = refinement_regions,
            max_length = max_length,
            ghost_layer_ratio = ghost_layer_ratio[1], # At least one margin cell in domain? valid block
            interior_point = interior_point,
            approximation_ratio = approximation_ratio,
            farfield_boundaries = hypercube_boundaries,
            verbose = verbose,
        )

        verbose && println("Obtaining KD tree...")
        centers = msh.centers |> permutedims
        spacing = msh.widths |> permutedims
        ndofs = size(centers, 1)
        tree = KDTree(centers')

        # calculate boundary distances and projections
        # considering families, not surfaces
        boundary_distances = Dict{String, AbstractVector{Float64}}()
        boundary_projs = Dict{String, AbstractMatrix{Float64}}()

        for (fam, famdef) in families
            if eltype(famdef) <: Tuple # hypercube boundary.
                # just fetch the already calculated projections
                projs = msh.boundary_projections[fam] |> permutedims
                indom = msh.boundary_in_domain[fam]

                sdfs = sqrt.(
                    sum(
                        (projs .- centers) .^ 2; dims = 2
                    ) |> vec
                ) .* (2 .* indom .- 1)

                boundary_projs[fam] = projs
                boundary_distances[fam] = sdfs
            else # STL boundary.
                # figure out the smallest SDF among surfaces
                minprojs = nothing
                minsdfs = nothing
                for sname in famdef
                    projs = msh.boundary_projections[sname] |> permutedims
                    indom = msh.boundary_in_domain[sname]

                    sdfs = sqrt.(
                        sum(
                            (projs .- centers) .^ 2; dims = 2
                        ) |> vec
                    ) .* (2 .* indom .- 1)

                    if isnothing(minsdfs)
                        minsdfs = sdfs
                        minprojs = projs
                    else
                        smaller = (sdfs .< minsdfs) |> findall

                        minprojs[smaller, :] .= projs[smaller, :]
                        minsdfs[smaller] .= sdfs[smaller]
                    end
                end

                boundary_projs[fam] = minprojs
                boundary_distances[fam] = minsdfs
            end
        end

        # figure out parts
        pranges = partition_ranges(ndofs, max_partition_cells)
        partitions = Partition{nd}[]
        for (ip, prange) in enumerate(pranges)
            verbose && println("Constructing partition $ip: range $prange")

            # start with image points
            image = collect(prange) |> x -> Int32.(x)
            icenters = centers[image, :]
            ispacing = spacing[image, :]

            # use interpolators in order to obtain domain
            domain = Int32[]
            for pt in stencil_points
                X = icenters .+ ispacing .* collect(pt)'

                intp = Interpolator(centers, X, tree; 
                    first_index = true, linear = false)

                domain = union(
                    domain, NNInterpolator.domain(intp)
                )
            end

            # same for boundary interpolators
            circumradius = sqrt.(
                sum(
                    ispacing .^ 2; dims = 2
                ) |> vec
            ) ./ 2

            boundaries = Dict{String, Boundary}()
            for fam in keys(families)
                boundaries[fam] = Boundary(
                    circumradius, icenters,
                    boundary_projs[fam][image, :], boundary_distances[fam][image],
                    tree, ghost_layer_ratio; Xc = centers
                )

                intp = boundaries[fam].image_interpolator
                domain = union(
                    domain, NNInterpolator.domain(intp)
                )
            end

            # re-index boundary interpolators
            hmap = NNInterpolator.index_map(domain)

            image_in_domain = map(i -> hmap[i], image)
            for (fam, bdry) in boundaries
                boundaries[fam] = Boundary(
                    NNInterpolator.filtered(bdry.image_interpolator, hmap),
                    bdry.points, bdry.normals, bdry.image_distances, bdry.ghost_distances,
                    image_in_domain[bdry.ghost_indices]
                )
            end

            # now build new interpolators for the entire domain
            pcenters = centers[domain, :]
            pspacing = spacing[domain, :]
            ptree = KDTree(pcenters')

            stencils = Dict{NTuple{nd, Int32}, Interpolator}()
            for pt in stencil_points
                X = pcenters .+ pspacing .* collect(pt)'

                intp = Interpolator(pcenters, X, ptree; 
                    first_index = true, linear = false)
                stencils[pt] = intp
            end

            push!(
                partitions,
                Partition(
                    domain, image, image_in_domain,
                    stencils, boundaries, pspacing, pcenters
                )
            )
        end

        # finally,  let's create surface structs 
        surface_dict = Dict{String, Surface}()
        for (sname, stl, L) in surfaces
            surface_dict[sname] = Surface(
                centers, spacing, tree, stl;
                max_length = L
            )
        end

        Domain(
            ndofs,
            msh,
            partitions,
            surface_dict,
            boundary_distances,
        )
    end

    """
    $TYPEDSIGNATURES

    Export surfaces and volume data to VTK file within a given folder.
    Kwargs are treated as field properties (cell data).
    """
    function export_vtk(
        folder::String,
        dom::Domain;
        include_volume::Bool = true,
        include_surface::Bool = true,
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

                grid = Mesher.STLHandler.stl2vtk(
                    joinpath(folder, sname), surf.stereolitography, vtm;
                    surf_data...)
            end

            Mesher.WriteVTK.vtk_save(vtm)
        end
    end

    include("arraybends.jl")
    using .ArrayBackends

    @declare_converter Interpolator
    @declare_converter Boundary
    @declare_converter Partition

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

    Backend "converters" can be used to convert arrays to certain
    array operation libraries (e.g. CUDA.jl) before operations are 
    conducted. Example:

    ```
    dom(
        u;
        conv_to_backend = CuArray, # may also be a custom function
        conv_from_backend = Array
    ) do part, udom
        @show typeof(udom) # CuArray
    end
    ```

    This ensures that, while operating with GPU parallelization, 
    information regarding a single partition at a time is ported to the GPU,
    thus satisfying far tighter memory requirements.
    """
    function (dom::Domain)(
        f, args::AbstractArray...; 
        conv_to_backend = nothing,
        conv_from_backend = nothing,
        kwargs...
    )
        ret = Vector{Any}(undef, length(dom.partitions))
        @threads for ip = 1:length(ret)
            part = dom.partitions[ip]

            ret[ip] = let pargs = map(
                a -> selectdim(a, 1, part.domain) |> copy, 
                args
            )
                mypart = part
                if !isnothing(conv_to_backend)
                    pargs = map(
                        a -> to_backend(a, conv_to_backend), pargs
                    )
                    mypart = to_backend(part, conv_to_backend)
                end

                r = f(mypart, pargs...; kwargs...)
            
                if !isnothing(conv_from_backend)
                    pargs = map(
                        a -> to_backend(a, conv_from_backend), pargs
                    )
                end

                for (a, pa) in zip(args, pargs)
                    selectdim(a, 1, part.image) .= selectdim(pa, 1, part.image_in_domain)
                end

                r
            end
        end

        ret
    end

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
                vimage .- vn .* ny
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
    Note that other field variable args. may be passed, even if the 
    BC function returns only a few return values at the boundary
    (which will be edited).

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

        if length(bdry.ghost_indices) == 0
            return
        end

        bargs = f(bdry, bdry.image_interpolator.(args)...; kwargs...)

        if !(bargs isa Tuple)
            if bargs isa AbstractVector && eltype(bargs) <: AbstractArray
                bargs = tuple(bargs...)
            else
                bargs = (bargs,)
            end
        end

        for (b, a) in zip(bargs, args)
            av = selectdim(
                a, 1, bdry.ghost_indices
            )

            if impose_at_ghost
                av .= b
            else
                gd = bdry.ghost_distances
                id = bdry.image_distances

                η = @. gd / id

                av .= η .* av .+ (1.0 .- η) .* b
            end
        end

        return
    end

    """
    $TYPEDSIGNATURES

    Serialize and save a domain
    """
    save_domain(fname::String, dom::Domain) = serialize(fname, dom)

    """
    $TYPEDSIGNATURES

    Load a domain
    """
    load_domain(fname::String) = deserialize(fname)

    """
    $TYPEDSIGNATURES

    Obtain stencil index `i` along dimension `dim` in block-structured array.
    `part(u, 0, 3, 0)` is equivalent to `ibm.getalong(part, u, 2, 3)`, for example
    """
    getalong(part::Partition, U::AbstractArray, dim::Int, i::Int) = let inds = zeros(
        Int64, dim
    )
        inds[end] = i
        part(U, inds...)
    end


    """  
    Evaluate the minmod flux limiter.
    """
    minmod(∇u::Real, Δu::Real) = min(abs(∇u), abs(Δu)) * abs(
        sign(∇u) + sign(Δu)
    ) / 2 / (abs(∇u) + 1e-10)
            
    """
    Evaluate MUSCL reconstruction at the left of face `i + 1/2`, as per
    minmod flux limiter
    """
    muscl_minmod(uim1::Real, ui::Real, uip1::Real) = (
          minmod(ui - uim1, uip1 - ui) * (ui - uim1) / 2 + ui
    )

    """
    $TYPEDSIGNATURES

    Evaluate MUSCL reconstruction at the left and right of face `i + 1/2`, as per
    minmod flux limiter. Operates on block-structured arrays.

    Runs along mesh dimension `dim`.

    Example:

    ```
    uL_i12, uR_i12 = ibm.MUSCL(partition, u, 2) # along dimension y
    ```
    """
    function MUSCL(part::Partition, u::AbstractArray, dim::Int64)

        uim1 = getalong(part, u, dim, -1)
        ui = getalong(part, u, dim, 0)
        uip1 = getalong(part, u, dim, 1)
        uip2 = getalong(part, u, dim, 2)

        uL = @. muscl_minmod(uim1, ui, uip1)
        uR = @. muscl_minmod(uip2, uip1, ui)

        (uL, uR)

    end

    """
    $TYPEDSIGNATURES

    Evaluate van-Albada flux limiter from a block-structured array of fluid properties.

    Runs along mesh dimension `dim`.

    Useful for debugging which points of your solution show first-order discretization.
    """
    function flux_limiter(part::Partition, u::AbstractArray, dim::Int64)

        uim1 = getalong(part, u, dim, -1)
        ui = getalong(part, u, dim, 0)
        uip1 = getalong(part, u, dim, 1)

        @. minmod(ui - uim1, uip1 - ui)

    end

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
    function smooth(part::Partition, u::AbstractArray)
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

    include("cfd.jl")
    using .CFD

    """
    $TYPEDSIGNATURES

    Obtain time step length for CFL = 1 at each cell given matrix of primitive
    variables (columns `p, T, u, v[, w]`). Returns a vector.
    """
    timescale(
        dom::Domain,
        fluid::CFD.Fluid,
        P::AbstractMatrix{Float64};
        conv_to_backend = nothing,
        conv_from_backend = nothing
    ) = let dt = Vector{Float64}(undef, length(dom))
        dom(
            P, dt;
            conv_to_backend = conv_to_backend,
            conv_from_backend = conv_from_backend
        ) do part, P, dt
            prims = eachcol(P)

            dt .= Inf64

            nd = length(prims) - 2

            a = CFD.speed_of_sound(fluid, prims[2])
            for i = 1:nd
                v = prims[2 + i]
                dx = @view part.spacing[:, i]

                @. dt = min(dt, dx / (abs(v) + a))
            end
        end

        dt
    end

    """
    $TYPEDSIGNATURES

    Run wall boundary condition on primitive variables.
    Imposes vel. gradient if specified, or laminar (Dirichlet 0) wall
    if `laminar = true`. Otherwise, uses an Euler wall.

    Velocity gradients may be given by passing function `du!dn` as a kwarg.
    The function should have the format:
    
    ```
    dV!dn = du!dn(bdry, V, P, args...; kwargs...)
    ```

    Where `V = |u|` is the non-normal velocity magnitude at the image points, 
    and the return value is its gradient normal to the wall.
    Note that all kwargs and extra args are passed to the function, as well as 
    the state variables.
    """
    function wall_bc(
        bdry::Boundary,
        P::AbstractMatrix{Float64},
        args::AbstractArray...;
        laminar::Bool = false,
        du!dn = nothing,
        fluid::CFD.Fluid,
        kwargs...
    )
        prims = eachcol(P)

        nd = length(prims) - 2
        if laminar # if laminar, zero at wall!
            for i = 1:nd
                u = prims[i + 2] 
                u .= 0.0
            end
        else # else, obtain normal component of velocity and remove it
            un = similar(prims[1])
            un .= 0.0

            for i = 1:nd
                u = prims[i + 2]
                n = @view bdry.normals[:, i]

                @. un += n * u
            end

            for i = 1:nd
                u = prims[i + 2]
                n = @view bdry.normals[:, i]

                @. u -= n * un
            end

            if !isnothing(du!dn) # if we have a non-zero velocity gradient, impose it
                V = similar(prims[1])
                V .= 0.0

                for i = 1:nd
                    u = prims[i + 2]
                    @. V += u ^ 2
                end

                @. V = sqrt(V)

                ϵ = eps(Float64)
                dV!dn = du!dn(bdry, V, P, args...; fluid = fluid, kwargs...)
                Vratio = @. (V - dV!dn * bdry.image_distances) / (V + ϵ)

                for i = 1:nd
                    u = prims[i + 2]
                    @. u *= Vratio # I'm using this ratio as a trick to impose the gradient
                    # in the direction of velocity
                end
            end
        end

        hcat(prims...)
    end

    """
    $TYPEDSIGNATURES

    Run freestream boundary condition on state variables.
    """
    function freestream_bc(
        bdry::Boundary,
        P::AbstractMatrix{Float64};
        freestream::CFD.Freestream
    )
        fluid = freestream.fluid
        free = freestream # an alias

        prims = eachcol(P)

        p = prims[1]
        T = prims[2]
        vels = prims[3:end]

        a = CFD.speed_of_sound(fluid, T)

        nd = length(vels)

        # obtain normal velocity to decide if we're at an inlet or outlet:
        un = similar(p)
        un .= 0.0

        for i = 1:nd
            u = vels[i]
            n = @view bdry.normals[:, i]

            @. un += n * u
        end

        # for temperature and velocities, use Dirichlet BC at inlet
        dirichlet = @. un > 0.0

        @. T = dirichlet * free.T + (1.0 - dirichlet) * T
        for i = 1:nd
            u = vels[i]
            uf = free.v[i]
            
            @. dirichlet * uf + (1.0 - dirichlet) * u
        end

        # for pressure, invert if subsonic
        @. dirichlet = dirichlet != (abs(un) < a)

        @. p = dirichlet * free.p + (1.0 - dirichlet) * p

        hcat(prims...)
    end

    include("LES.jl")
    using .LES

end
