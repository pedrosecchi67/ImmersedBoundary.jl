module ImmersedBoundary

    # includes stereolitography, linearalgebra, docstringextensions
    include("tree.jl")

    """
    $TYPEDFIELDS

    Struct to define a mesh stencil point
    """
    struct StencilPoint
        interpolator::Interpolator
        fetch_from::AbstractVector{Int64}
    end

    """
    $TYPEDSIGNATURES

    Constructor for a stencil point
    """
    function StencilPoint(
        tree::TreeCell,
        indices::Int...
    )

        indices = collect(indices)

        lvs = leaves(tree)
        centers = map(l -> l.center, lvs) |> x -> reduce(hcat, x)
        widths = map(l -> l.widths, lvs) |> x -> reduce(hcat, x)

        X = centers .+ widths .* indices

        fetch_from = zeros(Int64, length(lvs))
        interp_mask = falses(length(lvs))

        for (i, x) in enumerate(eachcol(X))
            c = find_leaf(tree, x)

            if isnothing(c)
                interp_mask[i] = true
            else
                if !(c.center ≈ x)
                    interp_mask[i] = true
                else
                    fetch_from[i] = c.index
                end
            end
        end

        fetch_mask = (@. !interp_mask)

        fetch_from[interp_mask] .= - (1:sum(interp_mask))
        X = X[:, interp_mask]

        intp = LinearInterpolator(tree, X)

        StencilPoint(intp, fetch_from)

    end

    """
    $TYPEDSIGNATURES

    Obtain values at a stencil point.
    """
    (stencil::StencilPoint)(v::AbstractVector) = let iv = stencil.interpolator(v)
        map(
            f -> (
                f < 0 ?
                iv[- f] :
                v[f]
            ),
            stencil.fetch_from
        )
    end

    """
    $TYPEDSIGNATURES

    Obtain values at a stencil point. The last index indicates the mesh point.
    """
    (stencil::StencilPoint)(v::AbstractArray) = mapslices(
        vv -> stencil(vv), v; dims = ndims(v)
    )

    _to_backend(converter, stencil::StencilPoint) = StencilPoint(
        Interpolator(
             converter(stencil.interpolator.stencils),
             converter(stencil.interpolator.weights),
        ), converter(stencil.fetch_from)
    )

    """
    $TYPEDFIELDS

    Struct to describe a mesh
    """
    struct Mesh
        tree::TreeCell
        stencils::Dict{Tuple{Vararg{Int64}}, StencilPoint}
        centers::Tuple{Vararg{AbstractArray{Float64}}}
        spacing::Tuple{Vararg{AbstractArray{Float64}}}
        interior_point::Vector{Float64}
        ncells::Int64
    end

    """
    $TYPEDSIGNATURES

    Obtain mesh from a tree.
    """
    Mesh(
        tree::TreeCell;
        interior_point = nothing
    ) = let stencils = Dict{Tuple{Vararg{Int64}}, StencilPoint}()
        lvs = leaves(tree)

        if isnothing(interior_point)
            interior_point = tree.origin .- 0.1 .* tree.widths
        end
        
        centers = map(
            i -> map(
                l -> l.center[i], lvs
            ),
            1:ndims(tree)
        ) |> Tuple
        spacing = map(
            i -> map(
                l -> l.widths[i], lvs
            ),
            1:ndims(tree)
        ) |> Tuple

        ncells = lvs |> length
        
        Mesh(
            tree,
            stencils,
            centers, spacing,
            interior_point,
            ncells,
        )
    end

    """
    $TYPEDSIGNATURES

    Obtain mesh by refining a tree to surfaces and 
    refinement regions.

    Example:

    ```
    stl1 = ibm.Stereolitography("stl1")
    stl2 = ibm.Stereolitography("stl2")

    msh = ibm.Mesh(
        [0.0, 0.0], [1.0, 1.0], # hypercube origin and widths
        stl1 => 0.01, # surface => local length pair
        stl2 => 0.01;
        refinement_regions = [ # pairs between distance functions and local lengths.
            ibm.Ball([0.0, 0.0], 0.1) => 0.05, # see ibm.Ball, ibm.Line, ibm.Box
            ibm.Ball([1.0, 0.0], 0.1) => 0.05
        ],
        clipping_surface = cat(stl1, stl2), # surface to clip the interior of the domain.
        interior_point = [0.0, 0.0] # reference point in the interior of the domain. Defaults to origin
    )
    ```
    """
    function Mesh(
        origin::AbstractVector{Float64}, widths::AbstractVector{Float64},
        surfaces::Pair{Stereolitography, Float64}...;
        refinement_regions = [],
        clipping_surface = nothing,
        interior_point = nothing,
        buffer_layer_depth::Int64 = 3,
        intersection_detection_ratio::Float64 = 1.1,
        cutting_ratio::Float64 = - 1.1,
        split_size::Int64 = 2,
        n_recursive_split::Int64 = 5,
    )

        tree = TreeCell(origin, widths; split_size = split_size)

        if n_recursive_split > buffer_layer_depth
            recursive_split!(tree, n_recursive_split - buffer_layer_depth)
        end

        refine!(
            tree, surfaces...;
            refinement_regions = refinement_regions,
            buffer_layer_depth = buffer_layer_depth,
            ratio = intersection_detection_ratio
        )

        set_numbering!(tree)
        if !isnothing(clipping_surface)
            clip_interior!(
                tree,
                clipping_surface;
                interior = interior_point,
                ratio = cutting_ratio,
            )
        end

        Mesh(
            tree; interior_point = interior_point,
        )

    end

    """
    $TYPEDSIGNATURES

    Obtain number of dimensions in a mesh.
    """
    Base.ndims(msh) = ndims(msh.tree)

    """
    $TYPEDSIGNATURES

    Obtain number of cells in a mesh
    """
    Base.length(msh::Mesh) = msh.ncells

    """
    $TYPEDSIGNATURES

    Export volume mesh to VTK file.
    Returns WriteVTK.jl object.

    All kwargs are converted to cell data.

    Example:

    ```
    vtk = ibm.vtk_grid("output", msh; u = rand(length(msh))) # exports output.vtu
    ibm.vtk_save(vtk)
    ```
    """
    WriteVTK.vtk_grid(fname, msh::Mesh; kwargs...) = octree2vtk(fname, msh.tree; kwargs...)

    """
    $TYPEDSIGNATURES

    Obtain the values of a vector/array at a stencil point.
    If a multi-dimensional array is passed, the last dimension
    is expected to refer to the mesh cell.
    """
    function (msh::Mesh)(v::AbstractArray, indices::Int64...)

        if !haskey(msh.stencils, indices)
            st = StencilPoint(msh.tree, indices...)

            bend = first(msh.centers) |> typeof |> x -> Base.typename(x).wrapper
            if !(bend <: Array)
                st = _to_backend(x -> bend(x), st)
            end

            msh.stencils[indices] = st
        end

        msh.stencils[indices](v)

    end

    """
    $TYPEDSIGNATURES

    ```
    v = getalong(u, msh, 2, -1)
    # ... is equivalent to:
    v = u[msh, 0, -1, 0]
    ```
    """
    function getalong(v::AbstractArray, msh::Mesh, dim::Int64, offset::Int64)

        inds = zeros(Int64, ndims(msh))
        inds[dim] = offset

        msh(v, inds...)

    end

    """
    $TYPEDSIGNATURES

    Obtain an interpolator from a mesh and a set of points (matrix columns)
    or to the face centers of a stereolitography object
    """
    Interpolator(
             msh::Mesh, X::Union{AbstractMatrix{Float64}, Stereolitography}
    ) = Interpolator(msh.tree, X)

    """
    $TYPEDSIGNATURES

    Obtain a linear interpolator from a mesh and a set of points (matrix columns)
    or to the face centers of a stereolitography object
    """
    LinearInterpolator(
             msh::Mesh, X::Union{AbstractMatrix{Float64}, Stereolitography}
    ) = LinearInterpolator(msh.tree, X)

    """
    $TYPEDSIGNATURES

    Convert mesh to a given array operation backend.

    Example:

    ```
    using CUDA
    import ImmersedBoundary as ibm

    msh = Mesh(#...

    msh = ibm.to_backend(cu, msh)

    # back to CPU:
    msh = ibm.to_backend(Array, msh)
    ```
    """
    function to_backend(converter, msh::Mesh)

        tree = msh.tree
        ncells = msh.ncells

        stencils = Dict(
            [
                k => _to_backend(converter, v) for (k, v) in msh.stencils
            ]...
        )

        Mesh(
            tree, stencils, 
            converter.(msh.centers), converter.(msh.spacing), msh.interior_point,
            ncells,
        )

    end

    """
    $TYPEDFIELDS

    Struct to describe a boundary.

    `distance_field` represents the distance of every cell in the mesh to the boundary.
    All other properties are defined for ghost cells only, however.
    """
    struct Boundary 
        ghost_indices::AbstractVector{Int64}
        distances::AbstractVector{Float64}
        image_distances::AbstractVector{Float64}
        normals::Tuple{Vararg{AbstractVector{Float64}}}
        centers::Tuple{Vararg{AbstractVector{Float64}}}
        projections::Tuple{Vararg{AbstractVector{Float64}}}
        distance_field::AbstractVector{Float64}
        image_interpolator::Interpolator
    end

    """
    $TYPEDSIGNATURES

    Obtain a boundary from a matrix of projections of each cell center upon the
    boundary, with the row index indicating the spatial dimension.

    Example:

    ```
    projections = rand(ndims(msh), length(msh)) # just to exemplify the shape

    bdry = Boundary(msh, projections)
    ```

    `ratio` defines that any cell separated from a boundary by no more than
    `ratio * norm(cell.widths) * sqrt(ndims(msh))` will be flagged as a ghost point.

    `isin` may be a vector of booleans with `true` for in-domain cells.
    """
    function Boundary(
        msh::Mesh, projections::AbstractMatrix{Float64}, isin = nothing;
        ratio::Real = 0.0,
    )

        if isnothing(isin)
            isin = trues(size(projections, 2))
        end

        ϵ = sqrt(eps(eltype(projections)))

        projections = eachrow(projections) |> Tuple

        distance_field = map((c, p) -> (c .- p) .^ 2, msh.centers, projections) |> sum |> x -> sqrt.(x)
        @. distance_field *= (2 * isin - 1)

        characteristic_lengths = map(w -> w .^ 2, msh.spacing) |> sum |> x -> sqrt.(x)
    
        normals = map(
            (c, p) -> (@. (c - p) / (distance_field + ϵ * sign(distance_field))),
            msh.centers, projections,
        )

        nd = ndims(msh)

        ghost_indices = (@. distance_field < ratio * characteristic_lengths * sqrt(nd)) |> findall

        select = v -> v[ghost_indices]

        distances = select(distance_field)
        normals = select.(normals)
        centers = select.(msh.centers)
        projections = select.(projections)

        image_distances = let cl = select(characteristic_lengths)
            @. max(
                   distances + cl * (sqrt(nd) * ratio),
                   cl * sqrt(nd)
            )
        end

        images = map(
            (p, n) -> p .+ n .* image_distances,
            projections, normals
        ) |> x -> mapreduce(v -> v', vcat, x)

        image_interpolator = LinearInterpolator(msh, images)

        Boundary(
            ghost_indices,
            distances, image_distances,
            normals, centers, projections,
            distance_field,
            image_interpolator
        )

    end

    """
    $TYPEDSIGNATURES

    Obtain boundary from stereolitography objects.

    `ratio` defines that any cell separated from a boundary by no more than
    `ratio * norm(cell.widths) * sqrt(ndims(msh))` will be flagged as a ghost point.
    """
    function Boundary(
        msh::Mesh, stls::Stereolitography...;
        ratio::Real = 0.0,
    )

        stl_joint = cat(stls...)
        stltree = STLTree(stl_joint)

        lvs = leaves(msh.tree)
        projs = map(
            l -> stltree(l.center)[1],
            lvs
        ) |> x -> reduce(hcat, x)
        isin = map(
            l -> !point_in_polygon(stltree, l.center; outside_reference = msh.interior_point),
            lvs
        )

        Boundary(msh, projs, isin; ratio = ratio)

    end

    """
    $TYPEDSIGNATURES

    Obtain boundary from a set of hypercube boundary specifications.

    Example for a standard, 3D CFD inlet boundary:

    ```
    inlet = Boundary(
        msh,
        (1, false), # "back" face, first (x) axis
        (2, false), # "bottom" face, second (y) axis
        (2, true), # "top" face, second (y) axis
        (3, false), # ...
        (3, true)
    )
    ```

    `ratio` defines that any cell separated from a boundary by no more than
    `ratio * norm(cell.widths) * sqrt(ndims(msh))` will be flagged as a ghost point.
    """
    function Boundary(
        msh::Mesh, faces::Tuple{Int64, Bool}...;
        ratio::Real = 1.1,
    )

        projs = nothing
        dists = nothing

        for (face, isfront) in faces
            if isnothing(projs)
                projs, dists = boundary_proj_and_dist(msh.tree, face, isfront)
            else
                p, d = boundary_proj_and_dist(msh.tree, face, isfront)

                for i = 1:length(d)
                    if d[i] < dists[i]
                        dists[i] = d[i]
                        projs[:, i] .= p[:, i]
                    end
                end
            end
        end

        Boundary(msh, projs; ratio = ratio)

    end

    """
    $TYPEDSIGNATURES

    Convert a boundary to a given vector handling backend.
    See the equivalent function for meshes.
    """
    to_backend(converter, bdry::Boundary) = Boundary(
        converter(bdry.ghost_indices),
        converter(bdry.distances), converter(bdry.image_distances),
        converter.(bdry.normals), converter.(bdry.centers), converter.(bdry.projections),
        converter(bdry.distance_field),
        Interpolator(
                     converter(bdry.image_interpolator.stencils),
                     converter(bdry.image_interpolator.weights),
        )
    )

    """
    $TYPEDFIELDS

    Struct to indicate a boundary condition.
    """
    struct BoundaryCondition
        f
        at_ghost::Bool
    end

    """
    $TYPEDSIGNATURES

    Obtain a boundary condition from a BC function.

    If `at_ghost` is true, the output values of `f` are directly assigned to
    ghost cells. Otherwise, they are interpolated between image points and the boundary.

    Example for non-penetration condition:

    ```
    bc = BoundaryCondition() do bdry, u, v
        nx, ny = bdry.normals

        unormal = @. nx * u + ny * v

        ( # note that fewer return values than input field variables may be specified.
            u .- nx .* unormal,
            v .- ny .* unormal
        )
    end
    ```
    """
    BoundaryCondition(f; at_ghost::Bool = false) = BoundaryCondition(f, at_ghost)

    """
    $TYPEDSIGNATURES

    Impose boundary condition (in place) to the provided variables.

    Example for non-penetration condition:

    ```
    bc = BoundaryCondition() do bdry, u, v # field variables interpolated to image points before imposition
        nx, ny = bdry.normals

        unormal = @. nx * u + ny * v

        ( # note that fewer return values than input field variables may be specified.
            u .- nx .* unormal,
            v .- ny .* unormal
        )
    end

    impose_bc!(bc, bdry, u, v)
    ```

    Kwargs are passed to the BC function.
    """
    function impose_bc!(
        bc, bdry::Boundary, u::AbstractVector{Float64}...;
        kwargs...
    )

        uimage = map(bdry.image_interpolator, u)
        ubdry = bc.f(bdry, uimage...; kwargs...)

        if bc.at_ghost
            for (uu, ub) in zip(u, ubdry)
                uu[bdry.ghost_indices] .= ub
            end
        else
            η = bdry.distances ./ bdry.image_distances

            for (uu, ub, ui) in zip(u, ubdry, uimage)
                uu[bdry.ghost_indices] .= (
                    @. η * ui + (1.0 - η) * ub
                )
            end
        end

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
    function centers_and_normals(stl::Stereolitography)

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
        stereolitography::Stereolitography
        points::Tuple{Vararg{AbstractVector{Float64}}}
        normals::Tuple{Vararg{AbstractVector{Float64}}}
        areas::AbstractVector{Float64}
        interpolator::Interpolator
    end

    """
    $TYPEDFIELDS

    Obtain a surface from a mesh and a stereolitography object.

    If `max_length` is provided, the STL surface is refined by tri splitting until no
    triangle side is larger than the provided value.

    If `linear` is set to true, any interpolation between field variables and the surface
    will be linear. Otherwise, Sherman's method (inverse distance weighing) is used.
    """
    function Surface(
            msh::Mesh, stl::Stereolitography, max_length::Float64 = 0.0; 
            linear::Bool = true
    )

        if max_length > 0.0
            stl = refine_to_length(stl, max_length)
        end

        nd = ndims(msh)

        interpolator = (
            linear ?
            LinearInterpolator(msh, stl) :
            Interpolator(msh, stl)
        )

        points = copy.(eachrow(stl.points)) |> Tuple

        _, normals = centers_and_normals(stl)

        normals = let point_normals = similar(stl.points)
            point_normals .= 0.0
            
            for ipts in eachrow(stl.simplices)
                for (n, ipt) in zip(eachcol(normals), ipts)
                    point_normals[:, ipt] .+= n ./ nd
                end
            end

            point_normals
        end

        ϵ = eltype(normals) |> eps |> sqrt
        areas = map(
                    n -> norm(n) + ϵ, eachcol(normals)
        )
        normals = copy.(eachrow(normals ./ areas')) |> Tuple

        Surface(
            stl,
            points,
            normals,
            areas,
            interpolator
        )

    end

    """
    $TYPEDSIGNATURES

    Interpolate a field property to a surface
    """
    (surf::Surface)(u::AbstractArray) = surf.interpolator(u)

    """
    $TYPEDSIGNATURES

    Obtain VTK grid for a given surface.

    Kwargs are taken as surface point data (see `getindex(u::AbstractArray, surf::Surface)`).
    If a provided array has more than `n` elements, if `n` is the number of points in the surface,
    the data is interpolated.
    """
    function surf2vtk(fname, surf::Surface; kwargs...)

        kws = Dict(
            [
                k => (
                    size(v, ndims(v)) == size(surf.stereolitography.points, 2) ?
                    v :
                    surf(v)
               ) for (k, v) in kwargs
            ]...
        )

        stl2vtk(fname, surf.stereolitography; kws...)

    end

    """
    $TYPEDSIGNATURES

    integrate a property throughout a surface
    """
    surface_integral(surf::Surface, u::AbstractVector) = let uu = (
        length(u) == size(surf.stereolitography.points, 2) ?
        u :
        u[surf]
    )
        surf.areas .* uu |> sum
    end

    """  
    Evaluate the van-Albada flux limiter.
    """
    van_albada(∇u::Real, Δu::Real) = ∇u * Δu / (Δu ^ 2 + ∇u ^ 2 + 1e-14) * abs(
        sign(∇u) + sign(Δu)
    )
            
    """
    Evaluate MUSCL reconstruction at the left of face `i + 1/2`, as per
    van-Albada's flux limiter
    """
    muscl_van_albada(uim1::Real, ui::Real, uip1::Real) = (
          van_albada(ui - uim1, uip1 - ui) * (ui - uim1) / 2 + ui
    )

    """
    $TYPEDSIGNATURES

    Evaluate MUSCL reconstruction at the left and right of face `i + 1/2`, as per
    van-Albada's flux limiter.

    Runs along mesh dimension `dim`.

    Example:

    ```
    uL_i12, uR_i12 = ibm.MUSCL(u, msh, 2) # along dimension y
    ```
    """
    function MUSCL(u::AbstractArray, msh::Mesh, dim::Int64)

        uim1 = getalong(u, msh, dim, -1)
        ui = getalong(u, msh, dim, 0)
        uip1 = getalong(u, msh, dim, 1)
        uip2 = getalong(u, msh, dim, 2)

        uL = @. muscl_van_albada(uim1, ui, uip1)
        uR = @. muscl_van_albada(uip2, uip1, ui)

        (uL, uR)

    end

    """
    $TYPEDSIGNATURES

    Evaluate van-Albada flux limiter from array of properties.

    Runs along mesh dimension `dim`.

    Useful for debugging which points of your solution show first-order discretization.
    """
    function flux_limiter(u::AbstractArray, msh::Mesh, dim::Int64)

        uim1 = getalong(u, msh, dim, -1)
        ui = getalong(u, msh, dim, 0)
        uip1 = getalong(u, msh, dim, 1)

        @. van_albada(ui - uim1, uip1 - ui)

    end

    """
    $TYPEDFIELDS

    Struct to define a coarse multigrid level
    """
    struct Multigrid
        clusters::AbstractVector{Tuple{Int64, Int64}}
        from_cluster::AbstractVector{Int64}
        size_ratios::AbstractVector{Float64}
    end

    """
    $TYPEDSIGNATURES

    Obtain a coarse multigrid level.

    The coarsened tree branches should have depth `max_depth` or lower.
    If not provided, the coarsest grid with equal-sized leaf cells is selected.
    """
    function Multigrid(msh::Mesh, max_depth::Int64 = 1000)

        tree = msh.tree

        blks = blocks(tree, max_depth)

        clusters = map(
                       blk -> let lvs = leaves(blk)
                           (
                                minimum(l -> l.index, lvs),
                                maximum(l -> l.index, lvs),
                           )
                       end,
            blks
        )

        size_ratios = zeros(length(msh))
        for blk in blks
            ratio = blk.split_size ^ depth(blk)

            for l in leaves(blk)
                size_ratios[l.index] = ratio
            end
        end

        from_cluster = zeros(Int64, length(msh))
        for (k, cluster) in enumerate(clusters)
                rng = cluster[1]:cluster[2]

                from_cluster[rng] .= k
        end

        Multigrid(clusters, from_cluster, size_ratios)

    end

    """
    $TYPEDSIGNATURES

    Convert a multigrid coarsener/prolongator struct to a given
    array backend.
    See the equivalent function for meshes.
    """
    to_backend(converter, mgrid::Multigrid) = Multigrid(
                                                        converter(mgrid.clusters),
                                                        converter(mgrid.from_cluster),
            converter(mgrid.size_ratios)
    )

    """
    $TYPEDSIGNATURES

    Convert an array to a given coarse grid level 
    (coarsen and prolongate back to the original).

    The reduction function is such that the value of u
    within a cluster will be replaced by `f(u[cluster])`.
    """
    (mgrid::Multigrid)(u::AbstractVector) = map(
        clst -> let v = 0.0
            for i = clst[1]:clst[2]
                v += u[i]
            end

            v / (clst[2] - clst[1] + 1)
        end,
        mgrid.clusters
    )[mgrid.from_cluster]

    
    """
    $TYPEDSIGNATURES

    Convert an array to a given coarse grid level 
    (coarsen and prolongate back to the original).
    """
    (mgrid::Multigrid)(u::AbstractArray) = mapslices(
        uu -> mgrid(uu), u; dims = ndims(u)
    )

    """
    Reshape an array to match the last dimension of another
    """
    _reshape_tolast(uref::AbstractVector, v::AbstractVector) = v
    _reshape_tolast(uref::AbstractArray, v::AbstractVector) = let s = ones(Int64, ndims(uref))
        s[end] = size(uref, ndims(uref))

        reshape(v, s...)
    end

    """
    $TYPEDSIGNATURES

    Obtain a backward derivative along dimension `dim`.
    """
    ∇(u::AbstractArray, msh::Mesh, dim::Int64) = let uim1 = getalong(u, msh, dim, -1)
        (u .- uim1) ./ _reshape_tolast(u, msh.spacing[dim])
    end

    """
    $TYPEDSIGNATURES

    Obtain a forward derivative along dimension `dim`.
    """
    Δ(u::AbstractArray, msh::Mesh, dim::Int64) = let uip1 = getalong(u, msh, dim, 1)
        (uip1 .- u) ./ _reshape_tolast(u, msh.spacing[dim])
    end

    """
    $TYPEDSIGNATURES

    Obtain a central derivative along dimension `dim`.
    """
    δ(u::AbstractArray, msh::Mesh, dim::Int64) = let uip1 = getalong(u, msh, dim, 1)
        (uip1 .- getalong(u, msh, dim, -1)) ./ (2 .* _reshape_tolast(u, msh.spacing[dim]))
    end

    """
    $TYPEDSIGNATURES

    Obtain the average of a property at face `i + 1/2` along dimension `dim`.
    """
    μ(u::AbstractArray, msh::Mesh, dim::Int64) = (
        getalong(u, msh, dim, 1) .+ u
    ) ./ 2

    include("cfd.jl")
    using .CFD

end

