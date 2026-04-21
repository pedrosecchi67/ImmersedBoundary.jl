module ImmersedBoundary

    using ThreadTools
    using Base.Threads: ReentrantLock, lock

    include("mesher.jl")
    using .BlockMesher
    using .BlockMesher.WriteVTK

    using .BlockMesher.LinearAlgebra
    using .BlockMesher.DocStringExtensions

    include("nninterp.jl")
    using .NNInterpolator
    using .NNInterpolator.ArrayAccumulator

    include("arraybends.jl")
    using .ArrayBackends

    include("point_implicit.jl")

    @declare_converter NNInterpolator.Accumulator
    
    export Stereolitography, refine_to_length, merge_points,
        Box, Ball, Line, DistanceField,
        feature_regions,
        export_vtk,
        Domain, impose_bc!,
        Interpolator, 
        ∇, Δ, δ, μ, MUSCL, laplacian_smoothing,
        getalong,
        advection, dissipation, divergent,
        surface_integral

    """
    $TYPEDFIELDS

    Struct to define a boundary
    """
    struct Boundary{Ti, Tf}
        ghost_indices::AbstractVector{Ti}
        ghost_distances::AbstractVector{Tf}
        image_distances::AbstractVector{Tf}
        points::AbstractMatrix{Tf}
        normals::AbstractMatrix{Tf}
        image_interpolator::NNInterpolator.Accumulator
    end

    @declare_converter Boundary

    """
    $TYPEDFIELDS

    Struct to define a mesh partition
    """
    struct Partition{Ti, Tf}
        id::Int
        margin::Int
        block_size::Int
        block_range::AbstractRange
        interpolator::NNInterpolator.Accumulator
        domain::AbstractVector{Ti}
        image::AbstractVector{Ti}
        image_in_domain::AbstractVector{Ti}
        centers::AbstractMatrix{Tf}
        spacing::AbstractMatrix{Tf}
        boundaries::Dict{String, Boundary{Ti, Tf}}
    end

    @declare_converter Partition

    """
    $TYPEDSIGNATURES

    Constructor for a boundary.
    """
    function Boundary(
            ghosts::AbstractVector{Ti}, projs::AbstractMatrix{Tf}, 
            centers::AbstractMatrix{Tf}, radii::AbstractVector{Tf},
            ptree::KDTree; ghost_layer_ratio::Real = 1.5f0
    ) where {Ti, Tf}
        ϵ = eps(Tf)

        normals = centers[ghosts, :] .- projs
        ghost_distances = sum(normals .^ 2; dims = 2) |> vec |> x -> sqrt.(x) .+ ϵ
        normals ./= ghost_distances

        image_distances = 2 .* radii[ghosts] .* ghost_layer_ratio

        image_interpolator = Interpolator(centers, projs .+ image_distances .* normals, ptree; 
                                          first_index = true, linear = true)
        NNInterpolator.ArrayAccumulator.change_data_types!(image_interpolator, Ti, Tf)

        Boundary{Ti, Tf}(
            ghosts,
            ghost_distances,
            image_distances,
            projs,
            normals,
            image_interpolator
        )
    end

    """
    $TYPEDSIGNATURES

    Constructor for a boundary.
    """
    function Boundary(
            msh::Mesh, bname::String,
            part::Partition{Ti, Tf}, ptree::KDTree; ghost_layer_ratio::Real = 1.5f0
    ) where {Ti, Tf}
        centers = part.centers
        radii = sum(part.spacing .^ 2; dims = 2) |> vec |> x -> sqrt.(x) ./ 2
        
        nd = size(centers, 2)

        dfield = msh.distance_fields[bname]

        should_check = falses(size(centers, 1))
        should_check[part.image_in_domain] .= true

        nperblock = length(should_check) ÷ length(part.block_range)

        ghosts = Ti[]
        projs = Vector{Tf}[]

        for (k, ib) in part.block_range |> enumerate
            brange = ((k - 1) * nperblock + 1):(k * nperblock)

            widths = msh.block_widths[:, ib]
            center = msh.block_origins[:, ib] .+ widths ./ 2

            R = norm(widths) * (part.margin * 2 + 2 * ghost_layer_ratio + part.block_size) / part.block_size / 2

            if dfield(center) > R * 1.2
                should_check[brange] .= false
            end
        end

        for i = 1:length(radii)
            if should_check[i]
                x = centers[i, :]
                r = radii[i] * 2 * ghost_layer_ratio

                p = BlockMesher.projection(dfield, x, r * 2)
                d = norm(p .- x)

                if d < r
                    push!(ghosts, Ti(i))
                    push!(projs, Tf.(p))
                end
            end
        end

        if length(projs) > 0
            projs = reduce(hcat, projs) |> permutedims
        else
            projs = Matrix{Tf}(undef, 0, nd)
        end

        Boundary(ghosts, projs, centers, radii, ptree;
            ghost_layer_ratio = ghost_layer_ratio)
    end

    """
    $TYPEDSIGNATURES

    Constructor for a boundary from hypercube faces.
    """
    function Boundary(
            msh::Mesh, part::Partition{Ti, Tf}, ptree::KDTree,
            faces::Tuple{Int, Bool}...; ghost_layer_ratio::Real = 1.5f0
    ) where {Ti, Tf}
        origin = msh.origin
        widths = msh.widths

        centers = part.centers
        radii = sum(part.spacing .^ 2; dims = 2) |> vec |> x -> sqrt.(x) ./ 2

        should_check = falses(size(centers, 1))
        should_check[part.image_in_domain] .= true

        ghosts = Ti[]
        projs = similar(centers)

        for i = 1:length(radii)
            if should_check[i]
                x = centers[i, :]
                d = Inf32
                p = copy(x)

                for (dim, front) in faces
                    _p = copy(x)
                    _p[dim] = (front ? origin[dim] + widths[dim] : origin[dim])

                    _d = norm(_p .- x)

                    if _d < d
                        d = _d
                        p .= _p
                    end
                end

                projs[i, :] .= p

                r = radii[i] * 2 * ghost_layer_ratio
                if d < r 
                    push!(ghosts, Ti(i))
                end
            end
        end

        projs = projs[ghosts, :]

        Boundary(ghosts, projs, centers, radii, ptree;
            ghost_layer_ratio = ghost_layer_ratio)
    end

    """
    $TYPEDSIGNATURES

    Build partition from mesh
    """
    function Partition(
        msh::Mesh, block_range::AbstractRange, margin::Int,
        X::AbstractMatrix, tree::KDTree;
        id::Int = 0, ghost_layer_ratio::Real = 1.5f0,
        hypercube_families = [],
    )
        centers, widths, is_margin = BlockMesher.get_cells(
            msh, block_range; margin = margin
        )
        centers = permutedims(centers)
        widths = permutedims(widths)

        nd = size(centers, 2)
        nblocks = length(block_range)
        ncells = size(centers, 1)
        ncells_total = size(msh.block_origins, 2) * (ncells ÷ nblocks)
        nperblock = msh.block_size ^ nd

        Tf = eltype(centers)
        Ti = (ncells_total > 1e9 ? Int64 : Int32)

        interp = Interpolator(X, centers, tree;
            first_index = true, linear = false)

        in_domain = @. !is_margin

        image_in_domain = findall(in_domain) |> x -> Ti.(x)
        image = map(
            block -> ((block - 1) * nperblock + 1):(block * nperblock),
            block_range
        ) |> x -> reduce(vcat, x) |> x -> Ti.(x)

        domain, hmap = NNInterpolator.domain(interp)
        NNInterpolator.re_index!(interp, hmap)
        NNInterpolator.ArrayAccumulator.change_data_types!(interp, Ti, Tf)

        part = Partition{Ti, Tf}(
            id, margin, msh.block_size, block_range,
            interp,
            domain, image, image_in_domain,
            centers, widths,
            Dict{String, Boundary{Ti, Tf}}()
        )

        let ptree = KDTree(centers')
            for bname in msh.distance_fields |> keys
                part.boundaries[bname] = Boundary(
                    msh, bname, part, ptree;
                    ghost_layer_ratio = ghost_layer_ratio
                )
            end

            for (bname, faces) in hypercube_families
                part.boundaries[bname] = Boundary(
                    msh, part, ptree, faces...;
                    ghost_layer_ratio = ghost_layer_ratio
                )
            end
        end

        part
    end

    """
    $TYPEDFIELDS

    Struct to define a surface for post-processing.
    `offsets` define the offset between property sampling points
    and the surface.
    """
    struct Surface{Ti, Tf}
        points::AbstractMatrix{Tf}
        offsets::AbstractVector{Tf}
        normals::AbstractMatrix{Tf}
        areas::AbstractVector{Tf}
        interpolator::NNInterpolator.Accumulator
        stl::Stereolitography
    end

    @declare_converter Surface

    """
    $TYPEDFIELDS

    Struct to define a domain
    """
    struct Domain{Ti, Tf}
        mesh::Mesh
        tree::KDTree
        partitions::Dict{Int, Partition{Ti, Tf}}
        surfaces::Dict{String, Surface{Ti, Tf}}
    end

    @declare_converter BlockMesher.Mesh
    @declare_converter Domain

    _index_range(
        N::Int, nmax::Int
    ) = let nparts = max(1, N ÷ nmax)
        psize = N ÷ nparts

        idxs = AbstractRange[]
        for i0 = 1:nparts
            push!(idxs, i0:nparts:N)
        end

        idxs
    end

    """
    $TYPEDSIGNATURES

    Construct a domain from a mesh.

    Defines partitions as per maximum partition size in cells (def. 100_000).
    Defines block margins at width `margin` cells (def. 2).

    `ghost_layer_ratio` (def. 1.5) defines a ratio between the width of the ghost
    cell layer and the local cell circumdiameter.

    Hypercube boundary families may be specified as:

    ```
    hypercube_families = [
        "inlet" => [
            (1, false), # x axis, front
            (2, false), # y axis, left
            (2, true), # y axis, right
            (3, false), # z axis, bottom
            (3, true) # z axis, top
        ],
        "outlet" => [(1, true)]
    ]
    ```
    """
    function Domain(
        msh::Mesh;
        margin::Int = 2,
        max_partition_size::Int = 100_000,
        ghost_layer_ratio::Real = 1.5f0,
        hypercube_families = [],
        verbose::Bool = false,
    )
        nd = size(msh.block_origins, 1)
        nblocks = size(msh.block_origins, 2)
        nperblock = (msh.block_size + margin * 2) ^ nd
        nblocks_perpart = max(max_partition_size ÷ nperblock, 1)

        ranges = _index_range(nblocks, nblocks_perpart)

        X = BlockMesher.get_cells(msh)[1]
        tree = KDTree(X)
        X = permutedims(X)

        n = 0
        lck = ReentrantLock()
        parts = Dict(
            tmap(
                id -> let r = ranges[id]
                    part = Partition(
                        msh, r, margin, X, tree; id = id, 
                        ghost_layer_ratio = ghost_layer_ratio,
                        hypercube_families = hypercube_families,
                    )
                    
                    verbose && lock(lck) do
                        n += 1
                        println("Built partition $n/$(length(ranges))")
                    end

                    id => part
                end, 1:length(ranges)
            )...
        )

        Ti = parts |> values |> first |> x -> eltype(x.domain)
        Tf = parts |> values |> first |> x -> eltype(x.centers)
        surfaces = Dict{String, Surface{Ti, Tf}}()

        Domain(
            msh, tree, parts, surfaces
        )
    end

    """
    $TYPEDSIGNATURES

    Obtain interpolator to a given set of points in a domain
    (matrix, size `(npts, ndims)`).

    Kwargs can specify if linear interpolation is used (def. `linear = true`)
    and the number of points for interpolation (def. `k = 2 ^ N`).
    """
    function NNInterpolator.Interpolator(
        dom::Domain{Ti, Tf}, Xc::AbstractMatrix{Tf2};
        linear::Bool = true, k::Int = 0, 
        _X::Union{AbstractMatrix{Tf}, Nothing} = nothing
    ) where {Ti, Tf, Tf2}
        tree = dom.tree
        if isnothing(_X)
            _X = zeros(Tf, length(dom), ndims(dom))

            dom(_X) do part, _X
                _X .= part.centers
            end
        end

        NNInterpolator.Interpolator(
            _X, Xc, tree; linear = linear, k = k,
            first_index = true
        )
    end

    """
    $TYPEDSIGNATURES

    Add surfaces to a domain.
    """
    function add_surfaces!(
        dom::Domain{Ti, Tf},
        surfaces::Pair{String, Stereolitography}...;
        ghost_layer_ratio::Real = 1.5f0
    ) where {Ti, Tf}
        ϵ = eps(Tf)
        tree = dom.tree
        
        h = zeros(Tf, length(dom))
        X = zeros(Tf, length(dom), ndims(dom))
        dom(h, X) do part, h, X
            h .= sum(part.spacing .^ 2; dims = 2) |> vec |> x -> sqrt.(x) .* ghost_layer_ratio
            X .= part.centers
        end

        for (sname, stl) in surfaces
            centers, normals = centers_and_normals(stl)

            centers = permutedims(centers)
            normals = permutedims(normals)

            areas = sum(normals .^ 2; dims = 2) |> vec |> x -> sqrt.(x) .+ ϵ
            normals ./= areas

            hs = let (idxs, _) = NNInterpolator.NearestNeighbors.nn(tree, centers')
                h[idxs]
            end

            points = centers .+ normals .* hs
            interp = Interpolator(dom, points;
                _X = X, k = ndims(dom) + 1)

            NNInterpolator.ArrayAccumulator.change_data_types!(interp, Ti, Tf)

            dom.surfaces[sname] = Surface{Ti, Tf}(
                points, hs, normals, areas, interp, deepcopy(stl)
            )

            delete!(dom.mesh.distance_fields, sname)
        end
    end

    """
    $TYPEDSIGNATURES

    Generate a domain given hypercube origins and widths.

    Surfaces should be specified as tuples with family names,
    `Stereolitography` structs and local refinement levels, respectively.

    Refinement regions, meanwhile, may be specified as tuples between distance
    functions (see `Line, Box, Ball, DistanceField` in this module) and local refinement levels.

    An approximate growth rate is accepted for the block octree/quadtree.
    The block size (along all axes) is given by `block_size` (def. 8).

    Example:

    ```
    stl = Stereolitography("wall.dat") # or STL in 3D
    stl2 = Stereolitography("wall2.dat")

    features = feature_regions(stl) |> DistanceField
    region2 = Stereolitography("region.stl") |> DistanceField

    msh = Domain(
        [-1.0, -1.0], [3.0, 3.0], # origin, widths
        ("wall", stl, 1e-3),
        ("wall2", stl2, 2e-3);
        growth_ratio = 2.0f0, # default
        refinement_regions = [
            features => 5e-4,
            region2 => 1e-2,
            Ball([0.0, 0.0], 1.0) => 2e-2,
            Box([-1.0, -1.0], [1.0, 2.0]) => 1e-2
        ],
    )
    ```

    Defines partitions as per maximum partition size in cells (def. 1_000_000).
    Defines block margins at width `margin` cells (def. 2).

    `ghost_layer_ratio` (def. 1.5) defines a ratio between the width of the ghost
    cell layer and the local cell circumdiameter.

    Hypercube boundary families may be specified as:

    ```
    hypercube_families = [
        "inlet" => [
            (1, false), # x axis, front
            (2, false), # y axis, left
            (2, true), # y axis, right
            (3, false), # z axis, bottom
            (3, true) # z axis, top
        ],
        "outlet" => [(1, true)]
    ]
    ```
    """
    function Domain(
        origin::AbstractVector, widths::AbstractVector,
        surfaces...;
        growth_ratio::Real = 2.0f0,
        tolerance::Real = 1f-7,
        block_size::Int = 8,
        refinement_regions::AbstractVector = [],
        margin::Int = 2,
        max_partition_size::Int = 1_000_000,
        ghost_layer_ratio::Real = 1.5f0,
        hypercube_families = [],
        verbose::Bool = false,
    )
        verbose && println("======DOMAIN GEN. PROCEDURE======")

        stls = Dict(
            [
                sname => stl for (sname, stl, _) in surfaces
            ]...
        )

        t0 = time()
        verbose && println("Generating region tree mesh...")

        dom = let msh = Mesh(
            origin, widths, surfaces...;
            growth_ratio = growth_ratio,
            tolerance = tolerance, block_size = block_size,
            refinement_regions = refinement_regions,
            verbose = verbose
        )
            verbose && println("[DONE] - $(time() - t0) seconds elapsed")

            nd, nblocks = size(msh.block_origins)
            verbose && println("""
$(nblocks) blocks
$(nblocks * block_size ^ nd) cells""")

            t0 = time()
            verbose && println("Partitioning and detecting boundary points...")

            dom = Domain(
                msh;
                max_partition_size = max_partition_size, margin = margin,
                ghost_layer_ratio = ghost_layer_ratio, 
                hypercube_families = hypercube_families,
                verbose = verbose
            )

            verbose && println("[DONE] - $(time() - t0) seconds elapsed")

            dom
        end

        t0 = time()
        verbose && println("Adding surfaces and removing distance field references...")

        add_surfaces!(dom, stls...; ghost_layer_ratio = ghost_layer_ratio)

        verbose && println("[DONE] - $(time() - t0) seconds elapsed")

        verbose && println("=================================")

        dom
    end

    """
    $TYPEDSIGNATURES

    Obtain values of field property array `u` at surface control points.
    """
    (surf::Surface)(u::AbstractArray) = surf.interpolator(u)

    """
    $TYPEDSIGNATURES

    Run function on all partitions of a domain.

    Example:

    ```
    domain(A, B) do part, A, B # here, A, B indicate arrays
        # selected to partition part, with padding for finite difference ops.

        # now we do whatever we want with them! We can edit them in-place, too

        r # return values are gathered in an array and returned.
    end
    ```

    In these arrays, the first index is always expected to correspond to the cell 
    index.

    Kwargs are passed as they are to the evaluation function.

    Conversion functions `conv_to_backend` and `conv_from_backend` may be passed to
    convert arrays (and partitions) to a custom array backend before any operations.

    Example:

    ```
    # for CuArrays:
    using CUDA

    conv_to_backend = x -> cu(x)
    conv_from_backend = x -> Array(x)
    ```

    `nthreads` may be specified to allow for multi-threading between partitions.
    """
    function (dom::Domain)(
        f,
        args::AbstractArray...; 
        conv_to_backend = identity,
        conv_from_backend = identity,
        nthreads::Int = 1,
        kwargs...
    )
        fp = part -> begin
            pargs = map(
                a -> selectdim(a, 1, part.domain) |> copy |> x -> to_backend(
                    x, conv_to_backend
                ), args
            )

            image = part.image
            part = to_backend(part, conv_to_backend)

            pargs = part.interpolator.(pargs)
            r = f(part, pargs...; kwargs...)

            for (a, pa) in zip(args, pargs)
                pa = selectdim(pa, 1, part.image_in_domain) |> copy
                
                selectdim(a, 1, image) .= to_backend(
                    pa, conv_from_backend
                )
            end

            r
        end

        tmap(
            fp, nthreads, values(dom.partitions)
        )
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
        args::AbstractArray...;
        impose_at_ghost::Bool = false,
        kwargs...
    )
        bdry = part.boundaries[bname]

        if length(bdry.ghost_indices) == 0
            return
        end

        intp = bdry.image_interpolator
        η = bdry.ghost_distances ./ bdry.image_distances
        ginds = bdry.ghost_indices

        iargs = intp.(args)
        gargs = map(
            a -> selectdim(a, 1, ginds), args
        )

        bargs = f(bdry, iargs...; kwargs...)

        if !(bargs isa Tuple)
            bargs = (bargs,)
        end

        for (ia, ga, ba) in zip(
            iargs, gargs, bargs
        )
            if impose_at_ghost
                ga .= ba
            else
                ga .= η .* ia .+ (1.0f0 .- η) .* ba
            end
        end
    end

    """
    $TYPEDSIGNATURES

    Get number of cells in domain
    """
    Base.length(dom::Domain) = let nd = size(dom.mesh.block_origins, 1)
        size(dom.mesh.block_origins, 2) * dom.mesh.block_size ^ nd
    end

    """
    $TYPEDSIGNATURES

    Get number of dimensions of domain
    """
    Base.ndims(dom::Domain) = size(dom.mesh.block_origins, 1)

    """
    $TYPEDSIGNATURES

    Create folder with name `fname` with multi-block VTK file.
    kwargs are exported as volume data.

    Only a given set of blocks may be exported if indices `block_indices`
        are specified.

    Flags `export_volume` and `export_surface` may be used to specify which
    forms of data will be exported.

    Additional data may be passed for surfaces using format:

    ```
    τ = rand(length(dom))
    
    # example:
    wall = dom.surfaces["wall"]

    surface_data = Dict(
        "wall" => wall(τ) # interpolating to boundary
    )
    ```
    """
    function export_vtk(
        fname::String, dom::Domain,
        block_indices = nothing; 
        surface_data::AbstractDict = Dict(),
        export_volume::Bool = true, export_surface::Bool = true,
        kwargs...
    )
        if isdir(fname)
            @warn "Overwriting output in folder $fname."
            rm(fname; recursive = true, force = true)
        end
        mkdir(fname)

        if export_volume
            vtk = vtk_grid(
                fname, dom.mesh, block_indices; _make_folder = false,
                kwargs...
            )
            vtk_save(vtk)
        end

        if export_surface
            vtm = vtk_multiblock(
                joinpath(fname, "SURFACE")
            )

            for (sname, surf) in dom.surfaces
                mydata = Dict{Symbol, AbstractArray}()

                for (k, v) in kwargs
                    mydata[k] = surf(v) |> BlockMesher._fix_export
                end

                if haskey(surface_data, sname)
                    t = surface_data[sname]

                    for p in propertynames(t)
                        v = getproperty(t, p)

                        mydata[p] = v |> BlockMesher._fix_export
                    end
                end

                vtk = vtk_grid(
                    joinpath(fname, sname), surf.stl, vtm;
                    mydata...
                )
                vtk_save(vtk)
            end

            vtk_save(vtm)
        end
    end

    """
    $TYPEDSIGNATURES

    Obtain value of field property array at position `i, j[, k]` 
    along block stencil. At block edges, periodic BCs are used within margins.
    An error is thrown if `i, j[, k]` exceeds margins.

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
        u::AbstractArray, ijk::Int...
    )
        if any(
            i -> abs(i) > part.margin, ijk
        )
            error("Attempting to access point $(ijk) beyond block margins (check argument `margin` in domain creation)")
        end

        nd = size(part.centers, 2)

        ublock = reshape(
            u, fill(part.block_size + 2 * part.margin, nd)..., :, 
            size(u)[2:end]...
        )

        @assert nd == length(ijk) "Number of dimensions specified in partition Cartesian grid call $(ijk) is incompatible with dimensionality"
        n_extra_dims = ndims(ublock) - length(ijk) - 1

        shift_dims = ((@. - ijk)..., 0, fill(0, n_extra_dims)...)

        circshift(ublock, shift_dims) |> x -> reshape(x, size(u)...)
    end

    """
    $TYPEDSIGNATURES

    Obtain stencil index `i` along dimension `dim` in a partition array.
    `part(u, 0, 3, 0)` is equivalent to `ibm.getalong(part, u, 2, 3)`, for example
    """
    getalong(part::Partition, U::AbstractArray, dim::Int, i::Int) = let inds = zeros(
        Int64, size(part.centers, 2)
    )
        inds[dim] = i
        part(U, inds...)
    end

    """
    $TYPEDSIGNATURES

    Obtain a backward derivative along dimension `dim`.
    """
    ∇(part::Partition, u::AbstractArray, dim::Int64) = let uim1 = getalong(part, u, dim, -1)
        (u .- uim1) ./ part.spacing[:, dim]
    end

    """
    $TYPEDSIGNATURES

    Obtain a forward derivative along dimension `dim`.
    """
    Δ(part::Partition, u::AbstractArray, dim::Int64) = let uip1 = getalong(part, u, dim, 1)
        (uip1 .- u) ./ part.spacing[:, dim]
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
        getalong(part, u, dim, 1) .+ u
    ) ./ 2

    """
    $TYPEDSIGNATURES

    Run iteration of laplacian smoothing (obtain average of neighbors)
    """
    function laplacian_smoothing(part::Partition, u::AbstractArray)
        uavg = similar(u)
        uavg .= 0

        cnt = 0
        for i = 1:size(part.spacing, 2)
            cnt += 2
            uavg .+= (
                getalong(part, u, i, -1) .+ getalong(part, u, i, 1)
            )
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

    """
    $TYPEDSIGNATURES

    Auxiliary function for `-∇⋅u`.
    Uses upwinding (`order = 1`) or linear-upwinding with MUSCL (`order = 2`).
    """
    function divergent(
        part::Partition, u::AbstractMatrix;
        order::Int = 2,
    )
        mdiv = similar(u, (size(u, 1),))
        mdiv .= 0

        for dim = 1:size(u, 2)
            v = @view u[:, dim]
            h = @view part.spacing[:, dim]

            vLim12 = vRim12 = vLip12 = vRip12 = nothing
            if order == 1
                vLim12 = getalong(part, v, dim, -1)
                vRim12 = v
                vLip12 = v
                vRip12 = getalong(part, v, dim, 1)
            elseif order == 2
                vim2 = getalong(part, v, dim, -2)
                vim1 = getalong(part, v, dim, -1)
                vip1 = getalong(part, v, dim, 1)
                vip2 = getalong(part, v, dim, 2)

                _, vLim12 = MUSCL(vim2, vim1, v)
                vRim12, vLip12 = MUSCL(vim1, v, vip1)
                vRip12, _ = MUSCL(v, vip1, vip2)
            else
                throw(error("Order $order unsupported for divergent"))
            end

            up_ip12 = @. (vLip12 + vRip12) / 2 >= 0.0
            up_im12 = @. (vLim12 + vRim12) / 2 >= 0.0

            # Godunov: no flow if in opposite directions
            has_flow_ip12 = @. !((vLip12 < 0.0) && (vRip12 > 0.0))
            has_flow_im12 = @. !((vLim12 < 0.0) && (vRim12 > 0.0))
            
            @. mdiv += (
                (
                    up_ip12 * vLip12 + (1 - up_ip12) * vRip12
                ) * has_flow_ip12 - (
                    up_im12 * vLim12 + (1 - up_im12) * vRim12
                ) * has_flow_im12
            ) / h
        end

        mdiv
    end

    """
    $TYPEDSIGNATURES

    Auxiliary function for `∇⋅(μ ∇ϕ)`.
    """
    function dissipation(
        part::Partition, μ::Union{Real, AbstractVector}, ϕ::AbstractArray
    )
        div = similar(ϕ)
        div .= 0

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
    $TYPEDSIGNATURES

    Obtain multigrid data structures from domain.
    """
    function PointImplicit.GeometricMultigrid.Multigrid(
        dom::Domain{Ti, Tf}, n_levels::Int
    ) where {Ti, Tf}
        X = zeros(Tf, length(dom), ndims(dom))
        V = zeros(Tf, length(dom))

        dom(X, V) do part, X, V
            V .= prod(part.spacing; dims = 2) |> vec
            X .= part.centers
        end

        mgrid = PointImplicit.GeometricMultigrid.Multigrid(
            X, n_levels, V
        )

        for (c, p) in zip(mgrid.coarseners, mgrid.prolongators)
            PointImplicit.GeometricMultigrid.ArrayAccumulator.change_data_types!(
                c, Ti, Tf
            )
            PointImplicit.GeometricMultigrid.ArrayAccumulator.change_data_types!(
                p, Ti, Tf
            )
        end

        mgrid
    end

    include("cfd.jl")
    using .CFD

    include("turbulence.jl")
    using .Turbulence

    include("ibl.jl")
    using .IBL

end
