module ImmersedBoundary

    include("mesher.jl")
    import .Mesher as mshr
    using .Mesher: Stereolitography, Ball, Box, Line,
        PlaneSDF, TriReference, Projection
    using .Mesher.WriteVTK
    using .Mesher.STLHandler: STLTree, point_in_polygon, 
        refine_to_length, stl2vtk

    using .Mesher.DocStringExtensions
    using .Mesher.LinearAlgebra

    include("nninterp.jl")
    using .NNInterpolator

    using Base.Iterators
    using Serialization

    """
    $TYPEDFIELDS

    Struct to define a Cartesian grid block
    """
    struct Block
        margin::Int32
        size::Tuple
        origin::Tuple
        widths::Tuple
        spacing::Tuple
    end

    """
    $TYPEDSIGNATURES

    Constructor for a block
    """
    Block(
            origin::AbstractVector{Float64}, widths::AbstractVector{Float64},
            size::Tuple{Vararg{Int}};
            margin::Int = 1
    ) = Block(
              margin, size, tuple(origin...), tuple(widths...), tuple((widths ./ size)...)
    )

    """
    $TYPEDSIGNATURES

    Obtain all indices that define positions in the block
    """
    cartesian_indices(blck::Block) = Base.product([1:(s + 2 * blck.margin) for s in blck.size]...)

    """
    $TYPEDSIGNATURES

    Obtain position of a cell center given its index
    """
    cell_center(blck::Block, i::Int...) = (@. blck.origin + blck.spacing * (i - 0.5 - blck.margin))

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
        ghost_ratios::Tuple{Float64, Float64}
    )
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
        sqnd = sqrt(size(normals, 2))
        image_distances = @. max(
            circumradius * sqnd, ghost_distances + circumradius * sqnd * 1.1
        )

        image_points = @. projs + normals * image_distances

        # construct image interpolator
        intp = Interpolator(X, image_points, tree; linear = true, first_index = true)

        Boundary(
            intp, ghosts, normals, 
            image_distances, ghost_distances,
            ghost_indices
        )
    end

    function _simplex_normal(simplex::Matrix{Float64})

        p0 = simplex[:, 1]

        if size(simplex, 1) == 2 # 2D
            dx = simplex[:, 2] .- p0

            return [
                dx[2], - dx[1]
            ]
        end

        u = simplex[:, 2] .- p0
        v = simplex[:, 3] .- p0

        cross(u, v) ./ 2

    end

    _simplex_center(simplex::Matrix{Float64}) = dropdims(
        sum(simplex; dims = 2); dims = 2
    ) ./ size(simplex, 2)

    """
    $TYPEDSIGNATURES

    Obtain simplex centers and normals (with norms equal to simplex areas).
    """
    function centers_and_normals(stl::Mesher.Stereolitography)

        simplices = map(
            simp -> stl.points[:, simp], eachcol(stl.simplices)
        )

        centers = reduce(
            hcat,
            map(
                _simplex_center, simplices
            )
        )
        normals = reduce(
            hcat,
            map(
                _simplex_normal, simplices
            )
        )

        (centers, normals)

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
    end

    """
    $TYPEDSIGNATURES

    Obtain a surface from a domain and a stereolitography object.

    If `max_length` is provided, the STL surface is refined by tri splitting until no
    triangle side is larger than the provided value.
    """
    function Surface(
        X::AbstractMatrix{Float64}, 
        tree::KDTree, 
        stl::Mesher.Stereolitography; max_length::Float64 = 0.0
    )

        if max_length > 0.0
            stl = refine_to_length(stl, max_length)
        end

        nd = size(stl.points, 1)
        points = permutedims(stl.points)

        interpolator = Interpolator(X, points, tree; first_index = true)

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

        Surface(
            deepcopy(stl),
            points,
            normals,
            areas,
            interpolator
        )

    end

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
    struct Partition
        size::Tuple
        block_size::Tuple
        spacing::AbstractMatrix{Float64}
        interpolator::Interpolator
        domain::AbstractVector{Int32}
        image::AbstractVector{Int32}
        in_domain::AbstractVector{Int32}
        blocks::AbstractVector{Block}
        boundaries::Dict{String, Boundary}
    end

    """
    $TYPEDFIELDS

    Struct defining a domain
    """
    struct Domain
        ndofs::Int64
        size::Tuple
        spacing::AbstractMatrix{Float64}
        blocks::AbstractVector{Block}
        boundary_distances::AbstractDict
        surfaces::AbstractDict
        partitions::AbstractVector{Partition}
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
        
    Generate a Building Cubes mesh defined by:

    * A hypercube origin;
    * A vector of hypercube widths;
    * A set of tuples in format `(name, surface, max_length)` describing
        stereolitography surfaces (`Mesher.Stereolitography`) and 
        the max. cell widths at these surfaces;
    * A "margin" of cells for each block, used for inter-block communication;
    * A block size, in the form of a tuple with number of cells along each axis, or
        an integer with the same number for all axes;
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
    * An interval of ratios between the cell circumradius and the SDF of a given surface for 
        which cells are defined as ghosts;
    * A point reference within the domain. If absent, external flow is assumed;
    * An approximation ratio between wall distance and cell circumradius past which
        distance functions are approximated; 
    * A maximum number of octree blocks per partition; and
    * A set of families defining surface groups for postprocessing, BC imposition and wall
        distance calculations.

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
    """
    function Domain(
            origin::Vector{Float64}, widths::Vector{Float64},
            surfaces::Tuple{String, Stereolitography, Float64}...;
            margin::Int64 = 2,
            block_sizes::Union{Tuple, Int} = 8,
            refinement_regions::AbstractVector = [],
            max_length::Float64 = Inf,
            ghost_layer_ratio::Tuple = (-2.1, 2.1),
            interior_point = nothing,
            approximation_ratio::Float64 = 2.0,
            verbose::Bool = false,
            max_partition_blocks::Int64 = 1000,
            families = nothing,
    )
        nd = length(origin)
        if block_sizes isa Number # if it's an int, use it for all dimensions
            block_sizes = tuple(
                fill(block_sizes, nd)...
            )
        end
        maxwidth = max(block_sizes...)

        if verbose
            println("Generating octree...")
        end
        msh = mshr.FixedMesh(
            origin, widths,
            [
                (bname, stl, s * maxwidth) for (bname, stl, s) in surfaces
            ]...;
            growth_ratio = 2.0 / (3 * sqrt(nd)) + 0.99,
            refinement_regions = [
                sdf => s * maxwidth for (sdf, s) in refinement_regions
            ],
            max_length = max_length * maxwidth,
            ghost_layer_ratio = -1.0 + ghost_layer_ratio[1] / maxwidth,
            interior_point = interior_point,
            approximation_ratio = approximation_ratio,
            verbose = verbose,
        )

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

        blocks = [ # define block structs as helpers
            Block(
                o, w, block_sizes; margin = margin
            ) for (o, w) in zip(
                eachcol(msh.origins), eachcol(msh.widths)
            )
        ]

        # array size for a few vars. we're storing in a minute
        array_size = (length(blocks), (block_sizes .+ 2margin)...)

        # where are the cells? are they actual cells, or blanked regions?
        in_domain = trues(array_size...)
        cell_centers = Array{Float64}(undef, array_size..., nd)
        for (ib, block) in enumerate(blocks)
            for inds in cartesian_indices(block)
                c = cell_center(block, inds...)

                cell_centers[ib, inds..., :] .= c

                if any(
                    (@. inds <= margin || inds > block.size + margin)
                )
                    in_domain[ib, inds...] = false
                end
            end
        end

        # now let's calculate SDFs and surface projections
        boundary_projs = Dict{String, AbstractArray{Float64}}()
        boundary_sdfs = Dict{String, AbstractArray{Float64}}()

        for (fam, famdef) in families
            if verbose
                println("Calculating SDFs to family $fam...")
            end

            bprojs = Array{Float64}(undef, array_size..., nd)
            bsdfs = Array{Float64}(undef, array_size...)

            boundary_projs[fam] = bprojs
            boundary_sdfs[fam] = bsdfs

            if all( # case with STL surface
                f -> (f isa String), famdef
            )
                stl = mapreduce( # join all STLs together
                    f -> stl_dict[f],
                    cat, famdef
                )
                tree = STLTree(stl) # construct triangulation tree structure

                for (ib, block) in enumerate(blocks) # iterate for blocks
                    # for the current octree cell, obtain the closest projection
                    c = msh.centers[:, ib]
                    p = zeros(nd)
                    sdf = Inf64
                    for surf in famdef
                        _p = msh.boundary_projections[surf][:, ib]
                        _sdf = (
                            msh.boundary_in_domain[surf][ib] * 2 - 1
                        ) * norm(
                            _p .- c
                        )

                        if _sdf < sdf
                            sdf = _sdf
                            p .= _p
                        end
                    end

                    circumradius = norm(block.widths) / 2

                    # if below threshold, calculate exact SDF. Else, use plane surface for
                    # approximation
                    triref = nothing
                    if abs(sdf) > circumradius * approximation_ratio
                        triref = TriReference(
                            PlaneSDF(
                                Projection(p, sdf >= 0.0),
                                c
                            ), 
                            c,
                            sdf >= 0.0
                        )
                    else
                        triref = TriReference(
                            tree,
                            c,
                            sdf >= 0.0
                        )
                    end

                    for inds in cartesian_indices(block)
                        c = cell_center(block, inds...) |> collect
                        p = Projection(triref, c)

                        bprojs[ib, inds..., :] .= p.projection
                        bsdfs[ib, inds...] = norm(
                            p.projection .- c
                        ) * (2 * p.interior - 1)
                    end
                end
            elseif all(
                f -> (f isa Tuple), famdef
            )
                bsdfs .= Inf64

                # this is simpler. Just use distance to plane for all hypercube faces
                for (dim, front) in famdef
                    ref = @. origin + widths / 2
                    proj = copy(ref)
                    if front
                        proj[dim] = origin[dim] + widths[dim]
                    else
                        proj[dim] = origin[dim]
                    end

                    triref = TriReference(
                        PlaneSDF(
                            Projection(proj, true), # we're always in the domain. It's an
                            ref # enveloping hypercube!
                        ), 
                        ref, true
                    )

                    for (ib, block) in enumerate(blocks)
                        for inds in cartesian_indices(block)
                            c = cell_center(block, inds...) |> collect
                            p = Projection(triref, c)
                            d = norm(p.projection .- c)

                            if d < bsdfs[ib, inds...]
                                bprojs[ib, inds..., :] .= p.projection
                                bsdfs[ib, inds...] = d
                            end
                        end
                    end
                end
            else
                throw(error("Erroneous family definition. Check docs"))
            end

            # eliminate blanked cells (further in than the ghost layer)
            # from the domain
            for (ib, block) in enumerate(blocks)
                circumradius = norm(block.spacing) / 2

                idom = selectdim(in_domain, 1, ib)
                sdfs = selectdim(bsdfs, 1, ib)
                
                let r = ghost_layer_ratio[1]
                    @. idom = idom && (sdfs >= r * circumradius)
                end
            end
        end

        # this is where it gets a little confusing, cause we have
        # to store tons of indices. Buckle up

        # indices of non-blanked cells in domain. Let's keep it in block-structured shape
        index_in_domain = reshape(
            cumsum(vec(in_domain)), size(in_domain)...
        )
        n_in_domain = index_in_domain[end] # number of non-blanked cells

        # positions of cell centers for non-blanked cells
        X_in_domain = let Xtot = reshape(cell_centers, :, nd)
            view(Xtot, vec(in_domain), :)
        end

        # KD tree for interpolator construction
        tree = KDTree(X_in_domain')

        # now, let's start partitioning the domain
        partitions = Partition[]
        pranges = partition_ranges(
            length(blocks), max_partition_blocks
        )
        for prange in pranges
            # slices for the current partition:
            indom = selectdim(in_domain, 1, prange)
            idx_indom = selectdim(index_in_domain, 1, prange)
            ccenters = selectdim(cell_centers, 1, prange)

            part_size = size(indom)

            # convert to array notation
            X = reshape(ccenters, :, nd)
            indom = vec(indom)
            idx_indom = vec(idx_indom)
            local_idx_indom = findall(indom) # indices of non-blanked cells among all in the 
            # current partition

            # obtain interpolator to current partition cells (including blanks)
            intp = Interpolator(
                X_in_domain,
                X,
                tree; 
                linear = false, first_index = true
            )
            dom = NNInterpolator.domain(intp) # obtain domain of influence for interpolated points

            # filter to non-blanked cells
            idx_indom = idx_indom[local_idx_indom]

            # convert interpolator and image indices to work on said domain of influence
            hmap = NNInterpolator.index_map(dom)
            intp = NNInterpolator.filtered(intp, hmap)
            idx_indom = map(i -> hmap[i], idx_indom)

            # calculate circumradius for each cell. Used for boundary definition
            circumradius = let cr = Array{Float64}(undef, part_size...)
                for (i, ib) in enumerate(prange)
                    selectdim(cr, 1, i) .= norm(blocks[ib].spacing) / 2
                end
                cr
            end
            # filter to non-blanked cells and obtain KD Tree
            circumradius = vec(circumradius)[local_idx_indom]
            X = X[local_idx_indom, :]

            part_tree = KDTree(X')

            # define boundaries
            boundaries = Dict{String, Boundary}()
            for bname in keys(boundary_projs)
                projs = selectdim(
                    boundary_projs[bname], 1, prange
                )
                sdfs = selectdim(
                    boundary_sdfs[bname], 1, prange
                )

                projs = reshape(projs, :, nd) |> x -> selectdim(x, 1, local_idx_indom)
                sdfs = vec(sdfs) |> x -> view(x, local_idx_indom)

                bdry = Boundary(
                    circumradius,
                    X,
                    projs, sdfs,
                    part_tree,
                    ghost_layer_ratio
                )

                boundaries[bname] = bdry
            end

            push!(
                partitions, Partition(
                    part_size,
                    blocks[1].size,
                    (msh.widths[:, prange] ./ block_sizes) |> permutedims,
                    intp,
                    dom,
                    idx_indom,
                    local_idx_indom,
                    blocks[prange],
                    boundaries
                )
            )
        end

        in_domain = vec(in_domain) |> findall # store indices of non-blanked cells, only

        # convert boundary SDFs to array notation
        for (bname, sdf) in boundary_sdfs
            boundary_sdfs[bname] = reshape(
                sdf, :
            )[in_domain]
        end

        # sinally,  let's create surface structs 
        surface_dict = Dict{String, Surface}()
        for (sname, stl, L) in surfaces
            surface_dict[sname] = Surface(
                X_in_domain, tree, stl;
                max_length = L
            )
        end

        Domain(
            n_in_domain,
            array_size,
            (msh.widths ./ block_sizes) |> permutedims,
            blocks,
            boundary_sdfs,
            surface_dict,
            partitions
        )
    end

    include("arraybends.jl")
    using .ArrayBackends

    @declare_converter Interpolator
    @declare_converter Boundary
    @declare_converter Partition

    """
    $TYPEDSIGNATURES

    Obtain values from an array in Cartesian grid format for a mesh partition.
    
    Example:

    ```
    u = rand(
        length(domain), 3, 2
    )

    domain(u) do part, udom
        U = part(udom)
        # shape (nblocks, nx, ny[, nz], 3, 2)
    end
    ```
    """
    function (part::Partition)(v::AbstractArray)
        extra_dims = size(v)[2:end]

        reshape(
            part.interpolator(v), part.size..., extra_dims...
        )
    end

    """
    $TYPEDSIGNATURES

    Impose values from a block-Cartesian array `v` upon the 
    correct indices of a flowfield array `vdom`
    """
    function update_partition!(part::Partition, vdom::AbstractArray, v::AbstractArray)
        extra_dims = size(vdom)[2:end]

        selectdim(vdom, 1, part.image) .= selectdim(
            reshape(v, :, extra_dims...), 1, part.in_domain
        );
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

        U = part(udom) # obtain Cartesian representation
        # of u. Shape (nblocks, nx, ny[, nz])
        R = part(rdom)

        # now do some Cartesian grid operations and
        # update R

        ibm.update_partition!(part, rdom, R) # send values to `rdom`
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
        conv_to_backend = CuArray,
        conv_from_backend = Array
    ) do part, udom
        @show typeof(udom) # CuArray
    end
    ```

    This ensures that, while operating with GPU parallelization, 
    information regarding a single partition at a time is ported to the GPU,
    thus satisfying far tighter memory requirements.
    """
    (dom::Domain)(
        f, args::AbstractArray...; 
        conv_to_backend = nothing,
        conv_from_backend = nothing,
        kwargs...
    ) = map(
        part -> let pargs = map(
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
                selectdim(a, 1, part.domain) .= pa
            end

            r
        end,
        dom.partitions
    )

    """
    $TYPEDSIGNATURES

    Impose boundary condition on block-structured array.

    Example for non-penetration condition:

    ```
    dom(u, v) do part, udom, vdom
        U = part(udom)
        V = part(vdom)

        ibm.impose_bc!(part, "wall", U, V) do bdry, uimage, vimage
            nx, ny = bdry.normals |> eachcol
            un = @. nx * uimage + ny * vimage

            (
                uimage .- un .* nx,
                vimage .- vn .* ny
            )
        end

        ibm.update_partition!(part, udom, U)
        ibm.update_partition!(part, vdom, V)
    end


    # alternative return value:
    uv = zeros(length(dom), 2)
    uv[:, 1] .= 1.0
    dom(uv) do part, uvdom
        UV = part(uvdom)
        
        ibm.impose_bc!(part, "wall", UV) do bdry, uvim
            uimage, vimage = eachcol(uvim)
            nx, ny = eachcol(bdry.normals)
            un = @. nx * uimage + ny * vimage

            uvim .- un .* bdry.normals
        end

        ibm.update_partition!(part, uvdom, UV)
    end
    ```

    Kwargs are passed directly to the BC function.
    Note that other field variable args. may be passed, even if the 
    BC function returns only a few return values at the boundary
    (which will be edited).
    """
    function impose_bc!(
        f,
        part::Partition, bname::String,
        args::AbstractArray{Float64}...;
        kwargs...
    )
        nd = length(part.size)

        pargs = map(
            a -> selectdim(
                let extra_dims = size(a)[(nd + 1):end]
                    reshape(a, :, extra_dims...)
                end, 1, part.in_domain
            ), args
        )
        bdry = part.boundaries[bname]

        if length(bdry.ghost_indices) == 0
            return
        end

        bargs = f(bdry, bdry.image_interpolator.(pargs)...; kwargs...)

        if !(bargs isa Tuple)
            if bargs isa AbstractVector && eltype(bargs) <: AbstractArray
                bargs = tuple(bargs...)
            else
                bargs = (bargs,)
            end
        end

        for (b, a) in zip(bargs, pargs)
            gd = bdry.ghost_distances
            id = bdry.image_distances
            η = @. gd / id

            av = selectdim(
                a, 1, bdry.ghost_indices
            )

            av .= η .* av .+ (1.0 .- η) .* b
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

    Clip margins from block-shaped array
    """
    view_clip_margins(blck::Block, a::AbstractArray) = let s = blck.size
        nd = length(s)
        i = [
            (
                i <= nd ?
                ((blck.margin + 1):(blck.margin + s[i])) :
                Colon()
            ) for i = 1:ndims(a)
        ]

        view(a, i...)
    end

    """
    $TYPEDSIGNATURES

    Export domain to VTK format.

    Builds a folder `fname` where all of the .vtr and .vtm files are stored.
    Kwargs are saved as cell data.
    """
    function export_vtk(
        fname::String,
        dom::Domain;
        include_volume::Bool = true,
        include_surface::Bool = true,
        kwargs...
    )
        # create directory
        if isdir(fname)
            @warn "Overwrite on output dir. $fname"
            rm(fname; recursive = true, force = true)
        end
        mkdir(fname)

        if include_volume
            vtm = vtk_multiblock(
                joinpath(fname, "volume")
            )

            for part in dom.partitions
                # for each part, obtain block-structured data
                part_kwargs = Dict(
                    [
                        k => selectdim(
                            v, 1, part.domain
                        ) |> part for (k, v) in kwargs
                    ]...
                )

                # create blocks
                for (ib, block) in enumerate(part.blocks)
                    vtk = vtk_grid(
                        vtm,
                        [
                            LinRange(o, o + w, s + 1) for (o, w, s) in zip(
                                block.origin, block.widths, block.size
                            )
                        ]...
                    )

                    for (k, v) in part_kwargs
                        vblock = view_clip_margins(
                            block, selectdim(v, 1, ib) |> copy
                        )
                        nd = length(block.size)
                        vblock = permutedims( # permute dimensions to match WriteVTK convention
                            vblock, (
                                ((nd + 1):ndims(vblock))..., (1:nd)...
                            )
                        )

                        vtk[String(k)] = vblock
                    end
                end
            end

            vtk_save(vtm)
        end

        if include_surface
            vtm = WriteVTK.vtk_multiblock(
                joinpath(fname, "surface")
            )

            for (sname, surf) in dom.surfaces
                surf_kwargs = Dict(
                    [
                        k => let vs = surf(v)
                            # if we have more than one dimension, reverse the first and last dimension (stl2vtk convention)
                            if ndims(vs) > 1
                                vs = permutedims(
                                    vs, ((2:ndims(vs))..., 1)
                                )
                            end

                            vs
                        end for (k, v) in kwargs
                    ]...
                )

                stl2vtk(
                    joinpath(fname, sname), surf.stereolitography, vtm;
                    surf_kwargs...
                )
            end

            vtk_save(vtm)
        end
    end

    """
    $TYPEDSIGNATURES

    Obtain a block-structured array with the cell centers of a partition.
    Returns an array of shape `(nblocks, nx, ny[, nz], ndims)`
    """
    cell_centers(part::Partition) = let centers = Array{Float64}(
        undef, part.size..., length(part.size) - 1
    )
        for (ib, block) in enumerate(part.blocks)
            for inds in cartesian_indices(block)
                centers[ib, inds..., :] .= cell_center(block, inds...)
            end
        end

        centers
    end

    """
    $TYPEDSIGNATURES

    Obtain values at a given stencil point given indices and a block-structured array.

    Example:

    ```
    u = rand(length(domain))
    ux = similar(u)

    domain(u, ux) do part, udom, uxdom
        U = part(udom)
        Ux = part(uxd)

        dx, dy = part.spacing |> eachcol

        Ux .= (
            part(U, 1, 0) .- part(U, -1, 0)
        ) ./ (2 .* dx)

        ibm.update_partition!(part, uxdom, Ux)
    end

    # ux is now the first, x-axis derivative of u
    ```
    """
    (part::Partition)(
        U::AbstractArray, inds::Int...
    ) = circshift(U, (0, (@. - inds)...))

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

    """
    $TYPEDSIGNATURES

    Obtain average of cell properties at all cell blocks in a partition.
    Receives block-structured array (shape `(nblocks, nx, ny[, nz], inds...)`)
    and returns another with collapsed grid dimensions (`(nblocks, 1, 1[, 1], inds...)`).

    This makes it easier to perform coarsening of block-structured data:

    ```
    Ucoarse .= ibm.block_average(part, U)
    ```
    """
    block_average(part::Partition, A::AbstractArray) = let nd = length(part.size)
        cnt = prod(part.size[2:end])
        
        sum(A; dims = 2:nd) ./ cnt
    end

    include("cfd.jl")
    using .CFD

    """
    $TYPEDSIGNATURES

    Obtain time step length for CFL = 1 at each cell. Returns a vector.
    """
    timescale(
        dom::Domain,
        fluid::CFD.Fluid,
        Q::AbstractMatrix{Float64};
        conv_to_backend = nothing,
        conv_from_backend = nothing
    ) = let dt = Vector{Float64}(undef, length(dom))
        dom(
            Q, dt;
            conv_to_backend = conv_to_backend,
            conv_from_backend = conv_from_backend
        ) do part, Qdom, dtdom
            dtblock = part(dtdom)
            Qblock = part(Qdom)

            prims = CFD.state2primitive(fluid, eachslice(Qblock; dims = ndims(Qblock))...)

            dtblock .= Inf64

            nd = length(prims) - 2

            a = CFD.speed_of_sound(fluid, prims[2])
            for i = 1:nd
                v = prims[2 + i]
                dx = @view part.spacing[:, i]

                @. dtblock = min(dtblock, dx / (abs(v) + a) / 2)
            end

            update_partition!(part, dtdom, dtblock)
        end

        dt
    end

    """
    $TYPEDSIGNATURES

    Run wall boundary condition on state variables.
    Imposes vel. gradient if specified, or laminar (Dirichlet 0) wall
    if `laminar = true`. Otherwise, uses an Euler wall.
    """
    function wall_bc(
        bdry::Boundary,
        Q::AbstractMatrix{Float64},
        du!dn::Union{Nothing, AbstractVector{Float64}} = nothing;
        laminar::Bool = false,
        fluid::CFD.Fluid
    )
        state = eachcol(Q)
        prims = CFD.state2primitive(fluid, state...)

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
                Vratio = @. (V - du!dn * bdry.image_distances) / (V + ϵ)

                for i = 1:nd
                    u = prims[i + 2]
                    @. u *= Vratio # I'm using this ratio as a trick to impose the gradient
                    # in the direction of velocity
                end
            end
        end

        CFD.primitive2state(fluid, prims...) |> x -> hcat(x...)
    end

    """
    $TYPEDSIGNATURES

    Run freestream boundary condition on state variables.
    """
    function freestream_bc(
        bdry::Boundary,
        Q::AbstractMatrix{Float64};
        freestream::CFD.Freestream
    )
        fluid = freestream.fluid
        free = freestream # an alias

        state = eachcol(Q)
        prims = CFD.state2primitive(fluid, state...)

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

        CFD.primitive2state(fluid, prims...) |> x -> hcat(x...)
    end

end