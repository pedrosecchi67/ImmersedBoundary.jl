module ImmersedBoundary

    include("mesher.jl")
    using .Mesher

    using .Mesher.DocStringExtensions
    using .Mesher.LinearAlgebra

    using NearestNeighbors

    """
    $TYPEDFIELDS

    Struct to hold an interpolator from mesh points
    """
    struct Interpolator
        n_outputs::Int64
        fetch_to::AbstractVector{Int64}
        fetch_from::AbstractVector{Int64}
        interpolate_to::AbstractVector{Int64}
        stencils::AbstractMatrix{Int64}
        weights::AbstractMatrix{Float64}
    end

    """
    $TYPEDSIGNATURES

    Obtain interpolator based on a KDTree and matrix of evaluation points.
    Uses linear interpolation (`linear = true`) or Sherman's interpolation (IDW).
    """
    function Interpolator(msh::Mesher.Mesh, X::AbstractMatrix, 
        tree::Union{KDTree, Nothing} = nothing;
        linear::Bool = true)

        if isnothing(tree)
            tree = KDTree(msh.centers)
        end

        n_outputs = size(X, 2)
        nd = size(X, 1)
        kneighs = 2 ^ nd

        if n_outputs == 0
            return Interpolator(
                0, Int64[], Int64[], 
                Int64[], 
                Matrix{Int64}(undef, kneighs, 0), Matrix{Float64}(undef, kneighs, 0)
            )
        end

        stencils, dists = knn(tree, X, kneighs)
        stencils = reduce(hcat, stencils)
        dists = reduce(hcat, dists)

        weights = similar(dists)

        for (j, x) in enumerate(eachcol(X))
            ds = @view dists[:, j]
            inds = @view stencils[:, j]
            cnts = @view msh.centers[:, inds]

            w = let ϵ = eps(eltype(ds))
                w = @. 1.0 / (ds + ϵ)
            end

            if linear
                A = mapreduce(
                    c -> [1.0 (c .- x)'],
                    vcat,
                    eachcol(cnts)
                ) .* w

                weights[:, j] .= pinv(A)[1, :] .* w
            else
                weights[:, j] .= (w ./ sum(w))
            end
        end

        threshold = sqrt(eps(eltype(dists)))
        is_same_point = @. abs(weights - 1.0) < threshold
        # find if all other weigths are zero
        let is_near_zero = map(
            c -> sum(abs.(c) .< threshold) == kneighs - 1,
            eachcol(weights)
        )
            is_same_point .*= is_near_zero'
        end

        should_fetch = map(any, eachcol(is_same_point))
        fetch_to = findall(should_fetch)
        fetch_from = vec(stencils)[vec(is_same_point)]
        @assert length(fetch_to) == length(fetch_from) "Coinciding mesh centers?"

        interpolate_to = findall(
            (@. !should_fetch)
        )

        Interpolator(
            n_outputs,
            fetch_to, fetch_from,
            interpolate_to, stencils[:, interpolate_to], weights[:, interpolate_to]
        )
    end

    """
    $TYPEDSIGNATURES

    Evaluate interpolator
    """
    function (intp::Interpolator)(Q::AbstractVector)
        Qnew = similar(Q, eltype(Q), intp.n_outputs)

        if length(intp.fetch_to) > 0
            Qnew[intp.fetch_to] .= Q[intp.fetch_from]
        end

        if length(intp.interpolate_to) > 0
            Qnew[intp.interpolate_to] .= dropdims(
                sum(
                    view(Q, intp.stencils) .* intp.weights;
                    dims = 1
                );
                dims = 1
            )
        end
        
        Qnew
    end

    """
    $TYPEDSIGNATURES

    Interpolate multi-dimensional array.
    The last dimension is assumed to refer to the cell index.
    """
    (intp::Interpolator)(Q::AbstractArray) = mapslices(
        intp, Q; dims = ndims(Q)
    )

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
        normals::AbstractMatrix{Float64}
        centers::AbstractMatrix{Float64}
        projections::AbstractMatrix{Float64}
        distance_field::AbstractVector{Float64}
        image_interpolator::Interpolator
    end

    """
    $TYPEDSIGNATURES

    Construct a boundary from a mesh, a boundary name, 
    a ratio between the min. image point distance and the ghost cell
    circumradius, and a range of distances/circumradii at which cells
    are detected to be ghost points.
    """
    function Boundary(
        msh::Mesher.Mesh, tree::KDTree, bname::String;
        ghost_distance_ratio::Tuple{Float64, Float64} = (- Inf64, 1.0),
        image_distance_ratio::Float64 = 1.0,
    )
        projections = msh.boundary_projections[bname]
        is_in_domain = msh.boundary_in_domain[bname]
        sdf = map(
            (p, c, i) -> norm(c .- p) * (2 * i - 1),
            eachcol(projections), eachcol(msh.centers), is_in_domain
        )

        ρmin, ρmax = ghost_distance_ratio
        image_distance_ratio = max(image_distance_ratio, ρmax)

        circumradii = map(norm, eachcol(msh.widths)) ./ 2
        ghost_indices = map(
            (d, r) -> (
                d >= ρmin * r && d <= ρmax * r
            ),
            sdf, circumradii
        ) |> findall

        centers = msh.centers[:, ghost_indices]
        projections = projections[:, ghost_indices]
        ghost_sdf = sdf[ghost_indices]
        circumradii = circumradii[ghost_indices]

        ϵ = eps(eltype(centers))
        nd = size(centers, 1)

        normals = (centers .- projections) .* (@. sign(ghost_sdf) / (abs(ghost_sdf) + ϵ))'
        image_distances = @. max(
            circumradii * sqrt(nd), circumradii * sqrt(nd) * image_distance_ratio + ghost_sdf
        )

        image_points = projections .+ normals .* image_distances'

        Boundary(
            ghost_indices,
            ghost_sdf, image_distances,
            normals, centers, projections,
            sdf, Interpolator(msh, image_points, tree)
        )
    end

    """
    $TYPEDFIELDS

    Struct to define a domain
    """
    struct Domain
        mesh::Mesher.Mesh
        tree::Union{Nothing, KDTree}
        stencil_interpolators::Dict{Tuple, Interpolator}
        boundaries::Dict{String, Boundary}
        centers::AbstractMatrix{Float64}
        widths::AbstractMatrix{Float64}
    end

    """
    $TYPEDSIGNATURES
    
    Instantiate domain from mesh.

    Boundaries are defined from a ratio between the min. image point distance 
    and the ghost cell circumradius, and a range of distances/circumradii at 
    which cells are detected to be ghost points.
    """
    function Domain(
        msh::Mesher.Mesh;
        ghost_distance_ratio::Tuple{Float64, Float64} = (- Inf64, 1.0),
        image_distance_ratio::Float64 = 1.0,
    )
        tree = KDTree(msh.centers)

        Domain(
            msh,
            tree,
            Dict{Tuple, Interpolator}(),
            Dict{String, Boundary}(
                [
                    bname => Boundary(
                        msh, tree, bname;
                        ghost_distance_ratio = ghost_distance_ratio,
                        image_distance_ratio = image_distance_ratio
                    ) for bname in keys(msh.boundary_in_domain)
                ]...
            ),
            copy(msh.centers), copy(msh.widths)
        )
    end

    """
    $TYPEDSIGNATURES

    Obtain value of array at stencil point.

    Example for y-axis derivative in 2D grid:

    ```
    dx, dy = eachrow(
        domain.widths
    )

    du!dy = (
        domain(u, 0, 1) .- domain(u, 0, -1)
    ) ./ (2 .* dy)
    ```
    """
    function (domain::Domain)(
        u::AbstractArray, inds::Int64...
    )
        v_inds = collect(inds)

        if all(v_inds .== 0)
            return copy(u)
        end

        if !haskey(domain.stencil_interpolators, inds)
            st = Interpolator(
                domain.mesh, 
                domain.mesh.centers .+ v_inds .* domain.mesh.widths,
                domain.tree
            )

            bend = domain.centers |> typeof |> x -> Base.typename(x).wrapper
            if !(bend <: Array)
                st = to_backend(st, x -> bend(x))
            end

            domain.stencil_interpolators[inds] = st
        end

        domain.stencil_interpolators[inds](u)
    end

    _mul_last(u::AbstractArray, v::AbstractVector) = let s = ones(Int64, ndims(u))
        s[end] = length(v)

        u .* reshape(v, s...)
    end
    _view_last(u::AbstractArray, i) = selectdim(
        u, ndims(u), i
    )

    """
    $TYPEDSIGNATURES

    Impose boundary conditions in-place for state variables in `args`.
    These state variables must be given by arrays in which the last dimension indicates
    the cell index.

    Function `f` should receive a boundary struct (boundary `domain.boundaries[bname]`)
    and the values of `args` interpolated to image points, and return the values of one or more
    state variables (as ordered in `args`) at the boundary.

    Example for a 2D Euler wall imposed on a velocity field:

    ```
    impose_bc!(domain, "wall", u, v) do bdry, ui, vi
        nx, ny = eachrow(bdry.normals)

        un = @. ui * nx + vi * ny

        (
            ui .- un .* nx,
            vi .- un .* ny
        )
    end
    ```

    Check `?ibm.Boundary` for other useful boundary properties.

    Another implementation with multidimensional arrays would be:

    ```
    impose_bc!(domain, "wall", UV) do bdry, UVi
        ui, vi = eachrow(UVi)

        nx, ny = eachrow(bdry.normals)

        un = @. ui * nx + vi * ny

        vcat(
            (ui .- un .* nx)',
            (vi .- un .* ny)'
        )
    end
    ```

    Note that fewer return values than `args` may be returned, case in which the remaining `args`
    are left unaltered, but are passed for boundary value calculation anyway.

    All kwargs are forwarded to `f`.

    The values are linearly interpolated/extrapolated to the boundary.
    For other applications, `f` may directly return values at ghost points by turning
    on flag `at_ghost`.
    """
    function impose_bc!(
        f, domain::Domain, bname::String, args::AbstractArray...;
        at_ghost::Bool = false,
        kwargs...
    )
        bdry = domain.boundaries[bname]

        aimage = map(bdry.image_interpolator, args)
        fa = f(bdry, aimage...; kwargs...)
        if !(fa isa Tuple)
            fa = (fa,)
        end

        η = bdry.distances ./ bdry.image_distances
        for (a, ab, ai) in zip(args, fa, aimage)
            if at_ghost
                _view_last(a, bdry.ghost_indices) .= ab
            else
                _view_last(a, bdry.ghost_indices) .= (
                    _mul_last(ai, η) .+ _mul_last(ab, 1.0 .- η)
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
        domain::Domain, stl::Mesher.Stereolitography; max_length::Float64 = 0.0
    )

        msh = domain.mesh

        if max_length > 0.0
            stl = Mesher.refine_to_length(stl, max_length)
        end

        nd = size(stl.points, 1)

        interpolator = Interpolator(msh, stl.points, domain.tree)

        points = copy(stl.points)

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
        normals = normals ./ areas'

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

    Obtain a surface from a domain and a set of mesh boundary names.

    If `max_length` is provided, the STL surface is refined by tri splitting until no
    triangle side is larger than the provided value.

    If no boundaries are selected, all available triangulated surfaces are used.
    """
    Surface(
        domain::Domain, bnames::String...; max_length::Float64 = 0.0
    ) = (
        length(bnames) == 0 ?
        Surface(
            domain, keys(domain.mesh.stereolitographies)...; max_length = max_length
        ) :
        let stl = map(
            bname -> domain.mesh.stereolitographies[bname], bnames
        ) |> x -> cat(x...)
            Surface(domain, stl; max_length = max_length)
        end
    )

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

        Mesher.stl2vtk(fname, surf.stereolitography; kws...)

    end

    """
    $TYPEDSIGNATURES

    integrate a property throughout a surface
    """
    surface_integral(surf::Surface, u::AbstractVector) = let uu = (
        length(u) == size(surf.stereolitography.points, 2) ?
        u :
        surf(u)
    )
        surf.areas .* uu |> sum
    end

    """
    $TYPEDSIGNATURES

    integrate a property throughout a surface. The last dimension in the array
    is assumed to refer to point/cell indices
    """
    surface_integral(surf::Surface, u::AbstractArray) = mapslices(
        uu -> surface_integral(surf, uu), u; dims = ndims(u)
    ) |> uu -> dropdims(uu; dims = ndims(uu))
    
    """
    $TYPEDSIGNATURES

    ```
    v = getalong(u, domain, 2, -1)
    # ... is equivalent to:
    v = domain(u, 0, -1, 0)
    ```
    """
    function getalong(v::AbstractArray, domain::Domain, dim::Int64, offset::Int64)

        inds = zeros(Int64, size(domain.centers, 1))
        inds[dim] = offset

        domain(v, inds...)

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
    minmod flux limiter.

    Runs along mesh dimension `dim`.

    Example:

    ```
    uL_i12, uR_i12 = ibm.MUSCL(u, domain, 2) # along dimension y
    ```
    """
    function MUSCL(u::AbstractArray, domain::Domain, dim::Int64)

        uim1 = getalong(u, domain, dim, -1)
        ui = getalong(u, domain, dim, 0)
        uip1 = getalong(u, domain, dim, 1)
        uip2 = getalong(u, domain, dim, 2)

        uL = @. muscl_minmod(uim1, ui, uip1)
        uR = @. muscl_minmod(uip2, uip1, ui)

        (uL, uR)

    end

    """
    $TYPEDSIGNATURES

    Evaluate van-Albada flux limiter from array of properties.

    Runs along mesh dimension `dim`.

    Useful for debugging which points of your solution show first-order discretization.
    """
    function flux_limiter(u::AbstractArray, domain::Domain, dim::Int64)

        uim1 = getalong(u, domain, dim, -1)
        ui = getalong(u, domain, dim, 0)
        uip1 = getalong(u, domain, dim, 1)

        @. minmod(ui - uim1, uip1 - ui)

    end

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
    ∇(u::AbstractArray, domain::Domain, dim::Int64) = let uim1 = getalong(u, domain, dim, -1)
        (u .- uim1) ./ _reshape_tolast(u, view(domain.widths, dim, :))
    end

    """
    $TYPEDSIGNATURES

    Obtain a forward derivative along dimension `dim`.
    """
    Δ(u::AbstractArray, domain::Domain, dim::Int64) = let uip1 = getalong(u, domain, dim, 1)
        (uip1 .- u) ./ _reshape_tolast(u, view(domain.widths, dim, :))
    end

    """
    $TYPEDSIGNATURES

    Obtain a central derivative along dimension `dim`.
    """
    δ(u::AbstractArray, domain::Domain, dim::Int64) = let uip1 = getalong(u, domain, dim, 1)
        (uip1 .- getalong(u, domain, dim, -1)) ./ (
            2 .* _reshape_tolast(u, view(domain.widths, dim, :))
        )
    end

    """
    $TYPEDSIGNATURES

    Obtain the average of a property at face `i + 1/2` along dimension `dim`.
    """
    μ(u::AbstractArray, domain::Domain, dim::Int64) = (
        getalong(u, domain, dim, 1) .+ u
    ) ./ 2

    """
    $TYPEDSIGNATURES

    Run iteration of laplacian smoothing (obtain average of neighbors)
    """
    function smooth(u::AbstractArray, domain::Domain)
        uavg = similar(u)
        uavg .= 0.0

        cnt = 0
        for i = 1:size(domain.centers, 1)
            cnt += 2
            uavg .+= (
                getalong(u, domain, i, -1) .+ getalong(u, domain, i, 1)
            )
        end

        uavg ./ cnt
    end

    """
    $TYPEDFIELDS

    Struct to accumulate values over variable-length stencils
    """
    struct Accumulator
        n_output::Int64
        stencils::Dict{Int64, Tuple}
    end

    """
    $TYPEDSIGNATURES

    Construct accumulator struct from stencils and weights.

    Example:

    ```
    acc = Accumulator(
        [[1, 2], [2, 3, 4]],
        [[-1.0, 2.0], [3.0, 4.0, 5.0]]
    )

    v = [1, 2, 3, 4]
    @show acc(v)
    # [3.0, 38.0]
    ```
    """
    function Accumulator(
        inds::AbstractVector,
        weights::Union{AbstractVector, Nothing} = nothing,
    )
        ls = length.(inds)

        d = Dict{Int64, Tuple}()
        for l in unique(ls)
            isval = (ls .== l) |> findall

            is = reduce(
                hcat, inds[isval]
            )
            ws = nothing
            if !isnothing(weights)
                ws = reduce(
                    hcat, weights[isval]
                )
            end

            d[l] = (isval, is, ws)
        end

        n = length(ls)
        Accumulator(n, d)
    end

    """
    $TYPEDSIGNATURES

    Run accumulator over vector
    """
    function (acc::Accumulator)(v::AbstractVector)
        vnew = similar(v, eltype(v), acc.n_output)

        vnew .= 0
        for (i, stencil, weights) in values(acc.stencils)
            if isnothing(weights)
                vnew[i] .= dropdims(
                    sum(
                        v[stencil];
                        dims = 1
                    );
                    dims = 1
                )
            else
                vnew[i] .= dropdims(
                    sum(
                        v[stencil] .* weights;
                        dims = 1
                    );
                    dims = 1
                )
            end
        end

        vnew
    end

    """
    $TYPEDSIGNATURES

    Run accumulator over array.
    Summation occurs over last dimension
    """
    (acc::Accumulator)(v::AbstractArray) = mapslices(
        acc, v; dims = ndims(v)
    )

    """
    $TYPEDSIGNATURES

    Obtain interpolator from source to destination domains.
    Uses cell volume weighing
    """
    function Interpolator(
        src::Domain, dst::Domain
    )::Accumulator
        idx, _ = nn(src.tree, dst.centers)

        circumradii = map(norm, eachcol(dst.widths)) ./ 2
        volumes = map(prod, eachcol(src.widths))

        indices = [
            inrange(src.tree, x, r) for (x, r) in zip(
                eachcol(dst.centers), circumradii
            )
        ]

        for k = 1:length(indices)
            if length(indices[k]) == 0
                indices[k] = [idx[k]]
            end
        end

        weights = [
            let v = volumes[inds]
                v ./ sum(v)
            end for inds in indices
        ]

        Accumulator(indices, weights)
    end

    include("cfd.jl")
    using .CFD

    include("arraybends.jl")
    using .ArrayBackends

    @declare_converter Interpolator
    @declare_converter Boundary
    @declare_converter Domain
    @declare_converter Surface
    @declare_converter Accumulator

end
