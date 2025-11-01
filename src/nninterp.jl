module NNInterpolator

    export Interpolator, KDTree

    using DocStringExtensions

    using LinearAlgebra
    using NearestNeighbors

    """
    $TYPEDFIELDS

    Struct to hold an interpolator from a point cloud
    """
    struct Interpolator
        n_outputs::Int32
        fetch_to::AbstractVector{Int32}
        fetch_from::AbstractVector{Int32}
        interpolate_to::AbstractVector{Int32}
        stencils::AbstractMatrix{Int32}
        weights::AbstractMatrix{Float64}
        first_index::Bool
    end

    """
    $TYPEDSIGNATURES

    Obtain interpolator based on a KDTree and matrix of evaluation points.
    Uses linear interpolation (`linear = true`) or Sherman's interpolation (IDW).

    If `first_index` is true, the first dimension is interpreted as the point index upon interpolation.
    Otherwise, the last dimension is interpreted as the point index (default).
    Note that, if `first_index` is true, then `Xc, X` will also be expected to have shape `(npts, ndims)`.

    If `n_neighbors` is not given, it is set to `2 ^ ndims`.
    """
    function Interpolator(Xc::AbstractMatrix, X::AbstractMatrix, 
        tree::Union{KDTree, Nothing} = nothing;
        linear::Bool = true,
        first_index::Bool = false,
        n_neighbors::Int = 0)

        if first_index
            Xc = permutedims(Xc)
            X = permutedims(X)
        end

        n_outputs = size(X, 2)
        nd = size(X, 1)
        kneighs = (
            n_neighbors == 0 ?
            2 ^ nd :
            n_neighbors
        )

        if n_outputs == 0
            return Interpolator(
                0, Int32[], Int32[], 
                Int32[], 
                Matrix{Int32}(undef, kneighs, 0), Matrix{Float64}(undef, kneighs, 0),
                first_index
            )
        end

        if isnothing(tree)
            tree = KDTree(Xc)
        end

        stencils, dists = knn(tree, X, kneighs)
        stencils = reduce(hcat, stencils)
        dists = reduce(hcat, dists)

        weights = similar(dists)

        for (j, x) in enumerate(eachcol(X))
            ds = @view dists[:, j]
            inds = @view stencils[:, j]
            cnts = @view Xc[:, inds]

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
            interpolate_to, stencils[:, interpolate_to], weights[:, interpolate_to],
            first_index
        )
    end

    """
    $TYPEDSIGNATURES

    Evaluate interpolator
    """
    function (intp::Interpolator)(Q::AbstractVector)
        Qnew = similar(Q, eltype(Q), intp.n_outputs)

        if length(intp.fetch_to) > 0
            view(Qnew, intp.fetch_to) .= view(Q, intp.fetch_from)
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
        ```
        function soft_mapslices(
            f, A::AbstractArray; dims::Int
        )
        ```

    Function to run mapslices in a non-allocating way by running `f`
    on all slices of `A` along dimension `dims`.
    """
    function soft_mapslices(
        f, A::AbstractArray; dims::Int
    )
        sliceinds = setdiff(1:ndims(A), dims) |> x -> tuple(x...)

        R = nothing
        Rslices = nothing
        for (i, aslice) in eachslice(A; dims = sliceinds) |> enumerate
            r = f(aslice)

            if isnothing(R)
                s = size(A) |> collect
                s[dims] = length(r)

                R = similar(A, s...)
                Rslices = eachslice(R; dims = sliceinds)
            end

            rslice = Rslices[i]
            rslice .= r
        end

        R
    end

    """
    $TYPEDSIGNATURES

    Interpolate multi-dimensional array.
    The last dimension is assumed to refer to the cell index.

    If `first_index` is true upon interpolator construction, 
    the first dimension is interpreted as the point index upon interpolation.
    Otherwise, the last dimension is interpreted as the point index (default).
    """
    (intp::Interpolator)(Q::AbstractArray) = (
        Q isa Array ? # raw Julia array
        mapslices(
            intp, Q; dims = (intp.first_index ? 1 : ndims(Q))
        ) : 
        soft_mapslices(
            intp, Q; dims = (intp.first_index ? 1 : ndims(Q))
        )
    )

    """
    $TYPEDSIGNATURES

    Obtain a new interpolator which only produces the values
    identified by indices in `i`.
    """
    function Base.getindex(intp::Interpolator, i)
        mask = falses(intp.n_outputs)
        mask[i] .= true

        fetch_remains = mask[intp.fetch_to] |> findall
        interpolate_remains = mask[intp.interpolate_to] |> findall

        new_inds = cumsum(mask)

        Interpolator(
            length(i),
            new_inds[intp.fetch_to[fetch_remains]],
            intp.fetch_from[fetch_remains],
            new_inds[intp.interpolate_to[interpolate_remains]],
            intp.stencils[:, interpolate_remains], intp.weights[:, interpolate_remains],
            intp.first_index
        )
    end

    """
    $TYPEDSIGNATURES

    Obtain domain from an interpolator (array of indices which influence the results)
    """
    domain(intp::Interpolator) = unique(
        [
            intp.fetch_from; vec(intp.stencils)
        ]
    )

    """
    $TYPEDSIGNATURES

    Convert a vector of indices in domain to a hashmap for re-indexing (see `filtered`)
    """
    function index_map(dom::AbstractVector)
        d = Dict{Int32, Int32}()

        i = 0
        for k in dom 
            if !haskey(d, k)
                i += 1
                d[k] = i
            end
        end

        d
    end

    """
    $TYPEDSIGNATURES

    Filter an interpolator to receive only the indices which are contained within its
    domain. Example:

    ```
    dom = NNInterpolator.domain(intp)
    intp_filter = NNInterpolator.filtered(intp, dom) # domain surmised if not provided

    @assert intp(v) ≈ intp_filter(v[dom])
    ```

    May also be used with `index_map` to avoid re-creating a hashmap of indices every
    time `filtered` is called for a different interpolator. Example:

    ```
    dom = [
        NNInterpolator.domain(intp1);
        NNInterpolator.domain(intp2)
    ] |> unique

    hmap = NNInterpolator.index_map(dom)

    # this saves memory!
    intp1 = NNInterpolator.filtered(intp1, hmap)
    intp2 = NNInterpolator.filtered(intp2, hmap)
    ```
    """
    function filtered(intp::Interpolator, 
        dom::Union{Nothing, AbstractVector, AbstractDict} = nothing)
        if isnothing(dom)
            dom = domain(intp)
        end

        if dom isa AbstractVector
            dom = index_map(dom)
        end

        Interpolator(
            intp.n_outputs,
            intp.fetch_to,
            map(i -> dom[i], intp.fetch_from),
            intp.interpolate_to,
            map(i -> dom[i], intp.stencils),
            intp.weights,
            intp.first_index
        )
    end

end
