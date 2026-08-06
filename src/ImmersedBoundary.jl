module ImmersedBoundary

    using Base.Threads: @threads, ReentrantLock, lock, nthreads
    using ThreadTools: tmap

    include("mesher.jl")
    using .BlockMesher

    using .BlockMesher.DocStringExtensions

    using .BlockMesher.LinearAlgebra
    using .BlockMesher.NearestNeighbors

    using .BlockMesher.WriteVTK

    include("nninterp.jl")
    using .NNInterpolator
    using .NNInterpolator.ArrayAccumulator

    include("cfd.jl")
    using .CFD
    include("turbulence.jl")
    using .Turbulence
    include("solver.jl")
    using .Solver

    include("arraybends.jl")
    using .ArrayBackends

    @declare_converter NNInterpolator.ArrayAccumulator.Accumulator

    @declare_converter CFD.FlowBC

    export Stereolitography, refine_to_length, merge_points,
        Box, Ball, Line, DistanceField,
        feature_regions, centers_and_normals,
        vtk_grid, vtk_save,
        Mesh, Domain

    using ProgressBars

    """
    $TYPEDSIGNATURES

    Get number of elements in mesh
    """
    Base.length(msh::Mesh) = msh.block_size ^ size(msh.block_origins, 1) * size(msh.block_origins, 2)

    """
    $TYPEDSIGNATURES

    Find vector of faces between neighbors in an octree.
    Each face is a tuple with elements:

    ```
    (
        dim, # face normal dimension
        own, # owner, index of cell to left
        neigh, # neighbor, index of cell to right
    )
    ```
    """
    function octree2faces(
        origins::AbstractMatrix{Tf}, widths::AbstractMatrix{Tf};
        verbose::Bool = false,
    ) where {Tf <: AbstractFloat}
        Ti = (size(origins, 2) > 1e9 ? Int64 : Int32)

        verbose && println("Running face detection...")

        centers = origins .+ widths ./ 2
        tree = KDTree(centers)
        radii = sum(widths .^ 2; dims = 1) |> vec |> x -> sqrt.(x) ./ 2

        faces = Tuple{Ti, Ti, Ti}[]
        lck = ReentrantLock()
        iter = 1:size(centers, 2)
        if verbose
            iter = ProgressBar(iter)
        end

        @threads for i = iter
            mins = origins[:, i] # intersect minima and maxima in cells
            maxs = mins .+ widths[:, i]
            neighs = inrange(tree, centers[:, i], radii[i] * 3.1f0)

            my_faces = Tuple{Ti, Ti, Ti}[]

            for j in neighs
                if i == j
                    continue
                end

                nmins = origins[:, j]
                nmaxs = nmins .+ widths[:, j]

                fo = max.(mins, nmins)
                fw = min.(maxs, nmaxs) .- fo

                tol = 0.01f0 * maximum(fw)
                n = 0 # count degenerate dimensions
                nz = 0
                for dim = 1:length(fw)
                    n += (fw[dim] < tol)
                    nz += fw[dim] < -tol
                end

                # not a face: not planar
                if n != 1 || nz > 0
                    continue
                end

                ndim = argmin(fw)
                if origins[ndim, j] < origins[ndim, i] # if left of cell, already registered
                    continue
                end

                push!(my_faces, (Ti(ndim), Ti(i), Ti(j)))
            end

            lock(lck) do 
                for f in my_faces
                    push!(faces, f)
                    if length(faces) % 10000 == 0
                        Base.GC.gc()
                    end
                end
            end
        end

        faces
    end

    """
    $TYPEDSIGNATURES

    In a similar format, find faces in an octree that neighbor
    hypercube boundaries.

    Returns faces as:

    ```
    (
        dim, # face normal dimension
        own, # owner, index of cell to left
        neigh, # neighbor, index of cell to right
    )
    ```
    """
    function hcube_faces(
        hcube_origins::AbstractVector{Tf}, hcube_widths::AbstractVector{Tf},
        origins::AbstractMatrix{Tf}, widths::AbstractMatrix{Tf}
    ) where {Tf <: AbstractFloat}
        Ti = (size(origins, 2) > 1e9 ? Int64 : Int32)

        faces = Tuple{Ti, Ti, Ti}[]
        for dim = 1:length(hcube_origins)
            idxs = findall(
                abs.(origins[dim, :] .- hcube_origins[dim]) .< widths[dim, :] .* 0.01f0
            )

            for i in idxs
                push!(
                    faces,
                    (Ti(dim), zero(Ti), Ti(i))
                )
            end

            idxs = findall(
                abs.(
                    origins[dim, :] .+ widths[dim, :] .- hcube_origins[dim] .- hcube_widths[dim]
                ) .< widths[dim, :] .* 0.01f0
            )

            for i in idxs
                push!(
                    faces,
                    (Ti(dim), Ti(i), zero(Ti))
                )
            end
        end

        faces
    end

    """
    $TYPEDSIGNATURES

    Find indices of ghost points and their projections on the surface, 
    given a distance field and column-major arrays of centers and widths.
    `ghost_layer_ratio` is a ratio between the ghost cell layer width and
    the local cell circumdiameter.
    """
    function ghosts_and_projections(
        dfield::DistanceField,
        centers::AbstractMatrix{Tf}, widths::AbstractMatrix{Tf};
        ghost_layer_ratio::Real = 1.5f0,
        verbose::Bool = false,
    ) where {Tf <: AbstractFloat}
        Ti = (
            size(centers, 2) < 1e9 ? Int32 : Int64
        )

        diams = sum(widths .^ 2; dims = 1) |> vec |> x -> sqrt.(x)
        verbose && println("Making initial screening...")
        ghosts = let tree = dfield.tree # first, simple KD-tree query
            _, dists = nn(tree, centers)
            dists .<= diams .* ghost_layer_ratio .* 2
        end |> findall
        ghosts = Ti.(ghosts)

        projs = similar(centers, (size(centers, 1), length(ghosts)))
        # go through potential ghost cells, calc. projections
        iterator = 1:length(ghosts)
        if verbose
            iterator = ProgressBar(iterator)
        end

        @threads for k in iterator
            g = ghosts[k]
            projs[:, k] .= BlockMesher.projection(dfield, centers[:, g], 
                diams[g] * ghost_layer_ratio * 2)
        end

        mask = let dists = sum((projs .- centers[:, ghosts]) .^ 2; dims = 1) |> vec |> x -> sqrt.(x)
            dists .<= diams[ghosts] .* ghost_layer_ratio
        end

        (ghosts[mask], projs[:, mask])
    end

    """
    $TYPEDSIGNATURES

    Find indices of ghost points and their projections on the surface, 
    given a distance field and a mesh.
    `ghost_layer_ratio` is a ratio between the ghost cell layer width and
    the local cell circumdiameter.
    """
    ghosts_and_projections(
        dfield::DistanceField, msh::Mesh;
        ghost_layer_ratio::Real = 1.5f0,
        verbose::Bool = false,
    ) = let (centers, widths, _) = (
        BlockMesher.get_cells(msh)
    )
        ghosts_and_projections(dfield, centers, widths; 
            ghost_layer_ratio = ghost_layer_ratio, verbose = verbose)
    end

    """
    $TYPEDSIGNATURES

    Obtain indices of ghost points and their projections on the surface, 
    given a vector of hypercube faces (tuples between dimension and
    front/back boolean)
    """
    function ghosts_and_projections(
        faces::AbstractVector{Tuple{Tib, Bool}},
        hcube_origin::AbstractVector, hcube_widths::AbstractVector,
        centers::AbstractMatrix{Tf}, widths::AbstractMatrix{Tf};
        ghost_layer_ratio::Real = 1.5f0, verbose::Bool = false,
    ) where {Tib <: Integer, Tf <: AbstractFloat}
        Ti = (
            size(centers, 2) < 1e9 ? Int32 : Int64
        )

        diams = sum(widths .^ 2; dims = 1) |> vec |> x -> sqrt.(x)
        verbose && println("Detecting hypercube boundary intersection...")
        verbose && begin
            for (dim, front) in faces
                println("Dimension $dim, $(front ? "front" : "back")")
            end
        end

        mask = falses(length(diams))
        projs = similar(centers)
        dists = similar(centers, (length(diams),))
        dists .= Inf32
        for (dim, front) in faces
            ps = copy(centers)
            ps[dim, :] .= (
                front ? hcube_origin[dim] + hcube_widths[dim] : hcube_origin[dim]
            )
            ds = sum(
                (ps .- centers) .^ 2; dims = 1
            ) |> vec |> x -> sqrt.(x)

            for (k, d) in enumerate(ds)
                if d < dists[k]
                    dists[k] = d
                    projs[:, k] .= ps[:, k]
                end

                if d < diams[k] * ghost_layer_ratio
                    mask[k] = true
                end
            end
        end

        ghosts = Ti.(findall(mask))
        projs = projs[:, ghosts]

        (ghosts, projs)
    end

    """
    $TYPEDSIGNATURES

    Obtain indices of ghost points and their projections on the surface, 
    given a vector of hypercube faces (tuples between dimension and
    front/back boolean)
    """
    ghosts_and_projections(
        faces::AbstractVector{Tuple{Ti, Bool}},
        msh::Mesh;
        ghost_layer_ratio::Real = 1.5f0, verbose::Bool = false,
    ) where {Ti <: Integer} = let (centers, widths, _) = (
        BlockMesher.get_cells(msh)
    )
        ghosts_and_projections(
            faces,
            msh.origin, msh.widths, centers, widths;
            ghost_layer_ratio = ghost_layer_ratio, verbose = verbose
        )
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
        offset_interpolator::NNInterpolator.Accumulator
        stl::Stereolitography
    end

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

    """
    $TYPEDSIGNATURES

    Obtain values of field property array `u` at surface control points.
    """
    (surf::Surface)(u::AbstractArray) = surf.interpolator(u)

    export at_offset
    """
    $TYPEDSIGNATURES

    Obtain values of field property array `u` at an offset from surface control points.
    """
    at_offset(surf::Surface, u::AbstractArray) = surf.offset_interpolator(u)

    """
    $TYPEDFIELDS

    Struct to define a domain partition
    """
    struct Partition{Ti <: Integer, Tf <: AbstractFloat}
        id::Int
        centers::AbstractMatrix{Tf}
        spacing::AbstractMatrix{Tf}
        face_accumulators::AbstractDict # Dict{Tuple{Int64, Bool}, Accumulator}
        face_owners_neighbors::AbstractDict # Dict{Int64, Tuple{AbstractVector{Ti}, AbstractVector{Ti}}}
        domain::AbstractVector{Ti}
        image::AbstractVector{Ti}
        image_in_domain::AbstractVector{Ti}
    end

    """
    $TYPEDSIGNATURES

    Obtain dimensionality of a domain partition
    """
    Base.ndims(part::Partition) = size(part.centers, 2)

    """
    $TYPEDSIGNATURES

    Struct to define a boundary
    """
    struct Boundary{Ti <: Integer, Tf <: AbstractFloat}
        ghost_indices::AbstractVector{Ti}
        projections::AbstractMatrix{Tf}
        normals::AbstractMatrix{Tf}
        image_distances::AbstractVector{Tf}
        ghost_distances::AbstractVector{Tf}
        image_interpolator::Accumulator
        image_domain::AbstractVector{Ti}
    end

    """
    $TYPEDSIGNATURES

    Constructor for a boundary from ghost point indices and their
    projections on the boundary
    """
    function Boundary(
        centers::AbstractMatrix{Tf}, widths::AbstractMatrix{Tf}, 
        tree::KDTree,
        ghost_indices::AbstractVector{Ti}, projs::AbstractMatrix{Tf};
        ghost_ratio::Real = 1.5f0,
    ) where {Ti <: Integer, Tf <: AbstractFloat}
        ghosts = @view centers[ghost_indices, :]
        normals = ghosts .- projs
        ghost_distances = sum(normals .^ 2; dims = 2) |> vec |> x -> sqrt.(x)
        normals ./= (ghost_distances .+ eps(Tf))
        
        image_distances = sum(widths[ghost_indices, :] .^ 2; dims = 2) |> vec |> x -> sqrt.(x) .* ghost_ratio .+ eps(Tf)
        images = projs .+ normals .* image_distances

        image_interpolator = Interpolator(
            centers, images, tree; first_index = true, linear = true,
        )

        dom, hmap = NNInterpolator.domain(image_interpolator)
        dom = Ti.(dom)
        NNInterpolator.re_index!(image_interpolator, hmap)

        Boundary{Ti, Tf}(
            ghost_indices, projs, normals,
            image_distances, ghost_distances, image_interpolator, dom
        )
    end

    """
    $TYPEDSIGNATURES

    Construct a partitioned boundary with a max. number of ghost points per
    partition.
    """
    function boundary_partitions(
        centers::AbstractMatrix{Tf}, widths::AbstractMatrix{Tf}, 
        tree::KDTree,
        ghost_indices::AbstractVector{Ti}, projs::AbstractMatrix{Tf},
        max_partition_size::Int = 100_000;
        ghost_ratio::Real = 1.5f0,
    ) where {Ti <: Integer, Tf <: AbstractFloat}
        bd = Dict{Int64, Boundary{Ti, Tf}}()

        for (ipart, indices) in enumerate(
            Base.Iterators.partition(1:length(ghost_indices), max_partition_size)
        )
            bd[ipart] = Boundary(
                centers, widths, tree,
                ghost_indices[indices], projs[indices, :];
                ghost_ratio = ghost_ratio
            )
        end

        bd
    end

    """
    $TYPEDFIELDS

    Struct to define a domain
    """
    struct Domain{Ti <: Integer, Tf <: AbstractFloat}
        ncells::Int
        mesh::Mesh
        partitions::AbstractDict # Dict{Int64, Partition{Ti, Tf}}
        boundaries::AbstractDict # Dict{String, Dict{Int64, Boundary{Ti, Tf}}}
        surfaces::AbstractDict # Dict{String, Surface{Ti, Tf}}
        reconstruction_kwargs::NamedTuple
    end

    """
    $TYPEDSIGNATURES

    Obtain dimensionality of a domain
    """
    Base.ndims(dom::Domain) = ndims(
        dom.partitions[1]
    )

    _averaging_weights(
        stencils::AbstractVector
    ) = map(
        s -> fill(1.0f0 / length(s), length(s)),
        stencils
    )

    """
    $TYPEDSIGNATURES

    Construct a domain from a mesh.

    Defines partitions as per maximum partition size in cells (def. 100_000).

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

    Domain partitions assume maximum cell numbers of `max_partition_size`
    and skirts of `partition_skirt_depth` cells.
    """
    function Domain(
        msh::Mesh;
        max_partition_size::Int = 100_000,
        partition_skirt_depth::Int = 2,
        ghost_layer_ratio::Real = 1.5f0,
        hypercube_families = [],
        verbose::Bool = false,
    )
        verbose && println("==Initiating domain definition procedure...==")

        nd = size(msh.block_origins, 1)
        nblocks = size(msh.block_origins, 2)
        block_size = msh.block_size

        ncells = block_size ^ nd * nblocks

        verbose && println("Working with $ncells cells")

        centers, widths, _ = get_cells(msh)
        origins = centers .- widths ./ 2

        verbose && println("Defining faces...")
        t0 = time()

        faces = [
            octree2faces(origins, widths; verbose = verbose);
            hcube_faces(msh.origin, msh.widths,
                origins, widths)
        ]
        nfaces = length(faces)

        Base.GC.gc()

        Ti = (
            max(nfaces, ncells) > 1e9 ?
            Int64 : Int32
        )
        Tf = Float32

        cells2faces = [
            Ti[] for _ = 1:ncells
        ]
        for (ifc, (_, o, n)) in enumerate(faces)
            if o != 0
                push!(cells2faces[o], Ti(ifc))
            end
            if n != 0
                push!(cells2faces[n], Ti(ifc))
            end
        end

        Base.GC.gc()

        verbose && println("[DONE] - $(time() - t0) seconds elapsed")
        verbose && println("$nfaces faces constructed")

        partitions = Dict{Int64, Partition{Ti, Tf}}()

        let partiter = Base.Iterators.partition(1:ncells, max_partition_size) |> collect
            nparts = length(partiter)
            verbose && println("Building $(nparts) partitions...")
            t0 = time()

            lck = ReentrantLock()

            idxiter = 1:nparts
            if verbose
                idxiter = ProgressBar(idxiter)
            end
            @threads for ipart = idxiter
                part = partiter[ipart]
                image = Ti.(collect(part))

                domain = Set{Ti}(image)
                for _ = 1:partition_skirt_depth
                    for c in collect(domain)
                        for f in cells2faces[c]
                            _, o, n = faces[f]

                            o != 0 && push!(domain, o) # owner
                            n != 0 && push!(domain, n) # neighbor
                        end
                    end
                end
                domain = collect(domain)
                sort!(domain)

                face_accumulators = Dict{Tuple{Int64, Bool}, Accumulator}()
                face_owners_neighbors = Dict{Int64, Tuple{AbstractVector{Ti}, AbstractVector{Ti}}}()
                let idx2domain = Dict(
                    d => Ti(k) for (k, d) in enumerate(domain)
                )
                    face_indices = reduce(union, cells2faces[domain]) |> unique

                    for dim = 1:size(centers, 1)
                        owners = Ti[]
                        neighbors = Ti[]
                        right_faces = [
                            Ti[] for _ = 1:length(domain)
                        ]
                        left_faces = [
                            Ti[] for _ = 1:length(domain)
                        ]

                        k = zero(Ti)
                        for f in face_indices
                            ndim, o, n = faces[f]

                            if ndim != dim
                                continue
                            end

                            o = (haskey(idx2domain, o) ? idx2domain[o] : zero(Ti))
                            n = (haskey(idx2domain, n) ? idx2domain[n] : zero(Ti))

                            add_left = true
                            add_right = true
                            if o == 0
                                o = n
                                add_right = false
                            end
                            if n == 0
                                n = o
                                add_left = false
                            end

                            push!(owners, o)
                            push!(neighbors, n)

                            k += 1
                            add_left && push!(left_faces[n], k)
                            add_right && push!(right_faces[o], k)

                            if k % 10000 == 0
                                Base.GC.gc()
                            end
                        end

                        face_owners_neighbors[dim] = (owners, neighbors)
                        face_accumulators[(dim, false)] = Accumulator(
                            left_faces,
                            _averaging_weights(left_faces);
                            first_index = true,
                        )
                        face_accumulators[(dim, true)] = Accumulator(
                            right_faces,
                            _averaging_weights(right_faces);
                            first_index = true,
                        )
                    end

                    image_in_domain = map(
                        i -> idx2domain[i], image
                    )

                    lock(lck) do
                        partitions[ipart] = Partition{Ti, Tf}(
                            ipart,
                            centers[:, domain] |> permutedims,
                            widths[:, domain] |> permutedims,
                            face_accumulators, face_owners_neighbors,
                            domain, image, image_in_domain,
                        )

                        Base.GC.gc()
                    end
                end
            end

            verbose && println("[DONE] - $(time() - t0) seconds elapsed")
        end

        verbose && println("Defining boundaries and surfaces...")
        t0 = time()

        boundaries = Dict{String, Dict{Int64, Boundary{Ti, Tf}}}()
        surfaces = Dict{String, Surface{Ti, Tf}}()
        let tree = KDTree(centers)
            diams = sum(widths .^ 2; dims = 1) |> vec |> x -> sqrt.(x)

            for (bname, faces) in hypercube_families
                println("Defining boundary $bname...")

                ghosts, projs = ghosts_and_projections(faces, msh;
                    ghost_layer_ratio = ghost_layer_ratio, verbose = verbose)
                projs = permutedims(projs)

                boundaries[bname] = boundary_partitions(
                    centers', widths', tree,
                    ghosts, projs, max_partition_size; ghost_ratio = ghost_layer_ratio
                )
            end

            for (bname, dfield) in msh.distance_fields
                println("Defining boundary $bname...")

                begin
                    ghosts, projs = ghosts_and_projections(dfield, msh;
                        ghost_layer_ratio = ghost_layer_ratio, verbose = verbose)
                    projs = permutedims(projs)

                    boundaries[bname] = boundary_partitions(
                        centers', widths', tree,
                        ghosts, projs, max_partition_size; ghost_ratio = ghost_layer_ratio
                    )
                end

                begin
                    stl = dfield.stl
                    fcenters, fnormals = centers_and_normals(stl)
                    idx, _ = nn(tree, fcenters)

                    h = diams[idx] .* ghost_layer_ratio # face offset defined by nearest
                    # cell
                    A = sum(fnormals' .^ 2; dims = 2) |> vec |> x -> sqrt.(x) .+ eps(Tf)
                    fnormals = fnormals' ./ A
                    fcenters = permutedims(fcenters)

                    bias = fnormals .* h
                    surfaces[bname] = Surface{Ti, Tf}(
                        fcenters, h,
                        fnormals, A,
                        Interpolator(centers', fcenters,
                            tree; first_index = true, bias = bias),
                        Interpolator(centers', fcenters .+ bias,
                            tree; first_index = true),
                        stl
                    )
                end
            end
        end

        verbose && println("[DONE] - $(time() - t0) seconds elapsed")

        verbose && println("==Done with domain definition!==")

        reconstruction_kwargs = (
            max_partition_size = max_partition_size,
            partition_skirt_depth = partition_skirt_depth,
            ghost_layer_ratio = ghost_layer_ratio,
            hypercube_families = deepcopy(hypercube_families),
        )

        Domain{Ti, Tf}(
            ncells,
            msh,
            partitions,
            boundaries,
            surfaces,
            reconstruction_kwargs,
        )
    end

    @declare_converter Partition
    @declare_converter Boundary
    @declare_converter Domain

    """
    $TYPEDSIGNATURES

    Run function through partitions.

    Example:

    ```
    domain(R, U) do part, r, u
        # do stuff on the partition, on arrays r and u
        # (you can change them in place)

        # selection of values belonging to the current partition
        # is done by indexing cells on the first dimension of the arrays

        # any return values are collected and returned by the 
        # global func. call
        k
    end
    ```

    Kwargs `conv_to_backend` and `conv_from_backend` should convert
    input arrays to and from custom backends (e.g. `x -> CuArray(x)`)
    for use with GPUs and the like.

    Other kwargs are passed to each function call.
    `nthreads` is set to the number of available threads by default.
    """
    function (dom::Domain{Ti, Tfd})(
        f,
        args::AbstractArray{Tf}...;
        conv_to_backend = nothing,
        conv_from_backend = nothing,
        n_threads::Int = 0,
        kwargs...
    ) where {Ti, Tfd, Tf}
        if n_threads == 0
            n_threads = nthreads()
        end

        @assert isnothing(conv_to_backend) == isnothing(conv_from_backend) "Backend converters must be provided at the same time"

        tmap(n_threads, keys(dom.partitions) |> collect) do i 
            let part = dom.partitions[i]
                dargs = map(
                    a -> selectdim(
                        a,
                        1, part.domain
                    ) |> copy, args
                )

                pimg_indom = part.image_in_domain
                pimg = part.image

                if !isnothing(conv_to_backend)
                    dargs = conv_to_backend.(dargs)
                    part = to_backend(part, conv_to_backend)
                end

                r = f(part, dargs...; kwargs...)

                if !isnothing(conv_from_backend)
                    dargs = conv_from_backend.(dargs)
                end

                for (a, da) in zip(args, dargs)
                    selectdim(a, 1, pimg) .= selectdim(da, 1, pimg_indom)
                end

                r
            end
        end
    end

    """
    $TYPEDSIGNATURES

    Get number of elements in a domain
    """
    Base.length(dom::Domain) = length(dom.mesh)

    export at_owners
    """
    $TYPEDSIGNATURES

    Obtain view to properties at face owners
    """
    at_owners(part::Partition{Ti, Tf}, u::AbstractArray, dim::Int) where {Ti, Tf} = selectdim(
        u, 1, part.face_owners_neighbors[dim][1]
    )

    export at_neighbors
    """
    $TYPEDSIGNATURES

    Obtain view to properties at face neighbors
    """
    at_neighbors(part::Partition{Ti, Tf}, u::AbstractArray, dim::Int) where {Ti, Tf} = selectdim(
        u, 1, part.face_owners_neighbors[dim][2]
    ) |> copy

    export at_faces
    """
    $TYPEDSIGNATURES

    Obtain properties at faces
    """
    function at_faces(
        part::Partition{Ti, Tf}, u::AbstractArray, dim::Int
    ) where {Ti, Tf}
        spown = at_owners(part, part.spacing, dim)
        spneigh = at_neighbors(part, part.spacing, dim)
        uown = at_owners(part, u, dim)
        uneigh = at_neighbors(part, u, dim)

        (uown .* spneigh[:, dim] .+ uneigh .* spown[:, dim]) ./ (
            spneigh[:, dim] .+ spown[:, dim]
        )
    end

    export green_gauss
    """
    $TYPEDSIGNATURES

    Obtain Green-Gauss integral over face properties
    """
    function green_gauss(
        part::Partition{Ti, Tf},
        uf::AbstractArray, dim::Int
    ) where {Ti, Tf}
        accl = part.face_accumulators[(dim, false)]
        accr = part.face_accumulators[(dim, true)]

        (accr(uf) .- accl(uf)) ./ part.spacing[:, dim]
    end

    export unsigned_green_gauss
    """
    $TYPEDSIGNATURES

    Obtain unsigned Green-Gauss integral over face properties
    """
    function unsigned_green_gauss(
        part::Partition{Ti, Tf},
        uf::AbstractArray, dim::Int
    ) where {Ti, Tf}
        accl = part.face_accumulators[(dim, false)]
        accr = part.face_accumulators[(dim, true)]

        (accr(uf) .+ accl(uf)) ./ part.spacing[:, dim]
    end

    export divergent
    """
    $TYPEDSIGNATURES

    Obtain Green-Gauss divergent over face properties
    """
    divergent(
        part::Partition{Ti, Tf},
        uf::Tuple
    ) where {Ti, Tf} = sum(
        dim -> green_gauss(part, uf[dim], dim),
        1:ndims(part)
    )

    export cell_gradient
    """
    $TYPEDSIGNATURES

    Obtain Green-Gauss gradient of a property at cell centers
    along dimension `dim`
    """
    function cell_gradient(
        part::Partition{Ti, Tf},
        u::AbstractArray, dim::Int
    ) where {Ti, Tf}
        green_gauss(
            part, at_faces(part, u, dim), dim
        )
    end

    """
    $TYPEDSIGNATURES

    Obtain Green-Gauss gradient of a property at cell centers.
    Returns tuple with gradients along each dimension
    """
    cell_gradient(
        part::Partition{Ti, Tf},
        u::AbstractArray
    ) where {Ti, Tf} = tuple(
        [
            cell_gradient(part, u, dim) for dim = 1:ndims(part)
        ]...
    )

    export face_distance
    """
    $TYPEDSIGNATURES

    Obtain distances between owners and neighbors at faces
    """
    function face_distance(
        part::Partition{Ti, Tf}, dim::Int
    ) where {Ti, Tf}
        spown = at_owners(part, part.spacing, dim)
        spneigh = at_neighbors(part, part.spacing, dim)

        (spown[:, dim] .+ spneigh[:, dim]) ./ 2
    end

    export owner_distance
    """
    $TYPEDSIGNATURES

    Obtain distance between face and owner cell center
    """
    function owner_distance(
        part::Partition{Ti, Tf}, dim::Int
    ) where {Ti, Tf}
        spown = at_owners(part, part.spacing, dim)

        spown[:, dim] ./ 2
    end

    export neighbor_distance
    """
    $TYPEDSIGNATURES

    Obtain distance between face and neighbor cell center
    """
    function neighbor_distance(
        part::Partition{Ti, Tf}, dim::Int
    ) where {Ti, Tf}
        spneigh = at_neighbors(part, part.spacing, dim)

        spneigh[:, dim] ./ 2
    end

    export face_gradient
    """
    $TYPEDSIGNATURES

    Obtain gradient of a property normal to a set of faces
    given values at cells
    """
    face_gradient(
        part::Partition{Ti, Tf}, u::AbstractArray, dim::Int
    ) where {Ti, Tf} = (
        at_neighbors(part, u, dim) .- at_owners(part, u, dim)
    ) ./ face_distance(part, dim)

    """
    $TYPEDSIGNATURES

    Obtain gradient of a property at faces
    given values and gradients at cells
    """
    function face_gradient(
        part::Partition{Ti, Tf}, 
        u::AbstractArray, ∇u::Tuple, dim::Int
    ) where {Ti, Tf}
        ∇uf = []
        for i = 1:ndims(part)
            if i == dim
                push!(
                    ∇uf, face_gradient(part, u, dim)
                )
            else
                push!(
                    ∇uf, at_faces(part, ∇u[i], dim)
                )
            end
        end

        tuple(∇uf...)
    end

    export JST_sensor
    """
    $TYPEDSIGNATURES

    Evaluate JST-type shock sensor at cells
    """
    function CFD.JST_sensor(
        part::Partition{Ti, Tf},
        p::AbstractArray, dim::Int = 0
    ) where {Ti, Tf}
        if dim == 0
            ν = similar(p); ν .= 1f-7

            for d = 1:ndims(part)
                ν .= max.(ν, JST_sensor(part, p, d))
            end

            return ν
        end

        face_diff = at_neighbors(part, p, dim) .- at_owners(part, p, dim)
        ν = (
            1f-7 .+ abs.(green_gauss(part, face_diff, dim))
        ) ./ (
            1f-7 .+ unsigned_green_gauss(part, abs.(face_diff), dim)
        )
    end

    @inline minmod(u1::Real, u2::Real) = min(abs(u1), abs(u2)) * (sign(u1) + sign(u2)) / 2

    export MUSCL
    """
    $TYPEDSIGNATURES

    Obtain MUSCL reconstruction at left and right sides of a face.
    Receives values at cells and (central scheme) gradients at cell centers.

    A Ducros-type shock sensor may be provided in kwarg `D`. If zero, a centered, 
    second-order scheme is used. If `high_order` is true, a fourth-order 
    Pade scheme substitutes it. 
    If one, MUSCL with the van-Leer limiter is used.
    """
    function MUSCL(
        part::Partition{Ti, Tf}, 
        u::AbstractArray, δu::AbstractArray, 
        dim::Int;
        D::Union{AbstractVector, Nothing} = nothing, high_order::Bool = false,
    ) where {Ti, Tf}
        down = owner_distance(part, dim)
        dneigh = neighbor_distance(part, dim)

        uown = at_owners(part, u, dim)
        uneigh = at_neighbors(part, u, dim)

        ∇uf = (uneigh .- uown) ./ (down .+ dneigh)
        δuo = at_owners(part, δu, dim)
        δun = at_neighbors(part, δu, dim)
        ∇u = (
            2 .* δuo .- ∇uf
        ) .* down
        Δu = (
            2 .* δun .- ∇uf
        ) .* dneigh

        @. ∇uf = minmod(Δu, ∇u) # re-use buffer

        uL, uR = (
            uown .+ ∇uf, uneigh .- ∇uf
        )

        if !isnothing(D)
            D = max.(
                at_owners(part, D, dim), at_neighbors(part, D, dim),
                1f-7,
            )

            uf = @. (uown * dneigh + uneigh * down) / (down + dneigh)
            if high_order
                @. uf += (δuo * down - δun * dneigh) / 8
            end

            @.  uL = uL * D + (1.0f0 - D) * uf
            @.  uR = uR * D + (1.0f0 - D) * uf
        end

        (uL, uR)
    end

    export impose_bc!
    """
    $TYPEDSIGNATURES

    Impose boundary conditions on arrays of field properties (cell identified
    by first index).

    Example for non-penetration condition in 2D:

    ```
    impose_bc!(dom, "wall", u, v) do bdry, u, v # values at image points
        nx, ny = bdry.normals |> eachcol
        # other assets:
        # * projections (matrix)
        # * image_distances (vector)

        un = @. nx * u + ny * v
        
        (
            u .- nx .* un,
            v .- ny .* un
        )
    end

    # alternative with single return value:
    impose_bc!(dom, "wall", uv) do bdry, uv
        n = bdry.normals
        un = sum(uv .* n; dims = 2) |> vec

        uv .- un .* n
    end
    ```

    Runs in `n_threads` threads, or all available.
    Kwargs `conv_to_backend` and `conv_from_backend`
    are similar to those seen in `Domain`.
    Other kwargs are passed to the BC function
    """
    function impose_bc!(
        f,
        dom::Domain{Ti, Tf}, bname::String,
        args::AbstractArray...; 
        n_threads::Int = 0,
        conv_to_backend = nothing,
        conv_from_backend = nothing,
        kwargs...
    ) where {Ti, Tf}
        parts = dom.boundaries[bname]

        if n_threads == 0
            n_threads = nthreads()
        end

        @assert isnothing(conv_to_backend) == isnothing(conv_from_backend) "Backend converters must be provided at the same time"

        tmap(
            n_threads, keys(parts) |> collect
        ) do ipart
            bdry = parts[ipart]

            ginds = bdry.ghost_indices
            η = bdry.ghost_distances ./ bdry.image_distances

            _args = args
            if !isnothing(conv_to_backend)
                bdry = to_backend(bdry, conv_to_backend)
                _args = to_backend(args, conv_to_backend)
            end

            iargs = map(_args) do a
                selectdim(a, 1, bdry.image_domain) |> bdry.image_interpolator
            end

            r = f(bdry, iargs...; kwargs...)
            if !(r isa Tuple)
                r = (r,)
            end

            if !isnothing(conv_from_backend)
                r = to_backend(r, conv_from_backend)
                iargs = to_backend(iargs, conv_from_backend)
            end

            for (a, ba, ia) in zip(args, r, iargs)
                ga = selectdim(a, 1, ginds)
                ga .= η .* ia .+ (1.0f0 .- η) .* ba
            end
        end
    end

    export export_vtk
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

    export multigrid
    """
    $TYPEDSIGNATURES

    Obtain from initial, fine domain, vector of coarse domains,
    vector of coarseners and vector of prolongators for multigrid.

    Coarseners and prolongators are callables such that `coarseners[i]`
    coarsens properties from level `i` to level `i - 1`, and the opposite
    is true of `prolongators[i]`.

    The interpolations are used as:

    ```
    coarse_doms, coarseners, prolongators = multigrid(dom)

    P = rand(length(dom), 5) # array, first index identifies cell

    Pc = coarseners[1](P)
    P .= prolongators[1](Pc)
    ```

    Check out `ImmersedBoundary.Solver.FAS!`!
    """
    function multigrid(dom::Domain{Ti, Tf}, max_levels::Int = 0; factor::Int = 2,
        verbose::Bool = false,) where {Ti, Tf}
        msh = dom.mesh

        _mdepth = log2(msh.block_size) |> floor |> Int64
        max_levels = (max_levels == 0 ? _mdepth : max_levels)

        coarse_doms = Domain[]
        coarseners = Accumulator[]
        prolongators = Accumulator[]

        coarsen_mesh = bsize -> Mesh(
            msh.origin, msh.widths, bsize, msh.block_origins, msh.block_widths, msh.distance_fields
        )

        Xold = Matrix{Tf}(undef, length(dom), ndims(dom)); Xold .= 0
        dom(Xold) do part, Xold
            Xold .= part.centers
        end
        tree_old = KDTree(Xold')

        bsize = msh.block_size
        for nit = 1:max_levels
            bsize = bsize ÷ factor

            verbose && println("Building domain for multigrid level $nit...")
            cdom = coarsen_mesh(bsize) |> msh -> Domain(msh; 
                verbose = verbose, dom.reconstruction_kwargs...)

            X = Matrix{Tf}(undef, length(cdom), ndims(cdom)); X .= 0
            cdom(X) do part, X
                X .= part.centers
            end
            tree = KDTree(X')

            verbose && println("Building coarsener and prolongator for level $nit...")
            coarsener = Interpolator(Xold, X, tree_old; first_index = true, linear = false)
            prolongator = Interpolator(X, Xold, tree; first_index = true, linear = false)

            push!(coarse_doms, cdom)
            push!(prolongators, prolongator)
            push!(coarseners, coarsener)

            tree_old = tree
            Xold = X

            Base.GC.gc()
        end
        tree_old = nothing; Xold = nothing;
        Base.GC.gc()

        (coarse_doms, prolongators, coarseners)
    end

end
