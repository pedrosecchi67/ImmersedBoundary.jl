module NNInterpolator

    using DocStringExtensions

    using LinearAlgebra
    using NearestNeighbors

    export Interpolator, KDTree, Multigrid

    module ArrayAccumulator

        using ..DocStringExtensions

        export Accumulator
        
        """
        $TYPEDFIELDS
        
        Struct to accumulate values over variable-length stencils
        """
        struct Accumulator
            n_output::Int64
            stencils::Dict{Int64, Tuple}
            first_index::Bool
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
        
        If `first_index` is true, the first array dimension is considered
        to be the summation axis.
        """
        function Accumulator(
            inds::AbstractVector,
            weights::Union{AbstractVector, Nothing} = nothing;
            first_index::Bool = false
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
            Accumulator(n, d, first_index)
        end
        
        """
        $TYPEDSIGNATURES
        
        Run accumulator over vector.

        If `Δ` is true, then the sum occurs over differences between the
        fetched stencil values, and the current stencil point.
        If `f` is provided, it is applied on the values to sum before adding them.

        Different reduction operations can be specified using `op`.
        """
        function (acc::Accumulator)(v::AbstractVector;
                Δ::Bool = false, f = identity,
                op = +)
            vnew = similar(v, eltype(v), acc.n_output)
        
            vnew .= 0
            for (i, stencil, weights) in values(acc.stencils)
                if isnothing(weights)
                    vnew[i] .= dropdims(
                        reduce(
                            op,
                            f(v[stencil]);
                            dims = 1
                        );
                        dims = 1
                    )
                else
                    vnew[i] .= dropdims(
                        reduce(
                            op,
                            (
                                Δ ?
                                f(v[stencil] .- v[i]') :
                                f(v[stencil])
                            ) .* weights;
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
        Summation occurs over last dimension if `first_index` is false,
        or the first if true.

        If `Δ` is true, then the sum occurs over differences between the
        fetched stencil values, and the current stencil point.
        If `f` is provided, it is applied on the values to sum before adding them.

        Different reduction operations can be specified using `op`.
        """
        (acc::Accumulator)(v::AbstractArray;
                Δ::Bool = false, f = identity,
                op = +) = mapslices(
            vv -> acc(vv; Δ = Δ, f = f, op = op), v; dims = (acc.first_index ? 1 : ndims(v))
        )
        
    end
    using .ArrayAccumulator

    """
    Obtain linear interpolation weights
    """
    function linear_weights(
        X::AbstractMatrix, 
        indices::AbstractVector,
        x::AbstractVector
    )
        ϵ = sqrt(
            eps(eltype(X))
        )

        dX = X[:, indices] .- x
        
        distances = sum(
            dX .^ 2; dims = 1
        ) |> vec |> x -> sqrt.(x) .+ ϵ

        w = 1.0 ./ distances
        w = let A = [
            dX' ones(size(dX, 2))
        ]
            pinv(A .* w)[end, :] .* w
        end

        mask = @. abs(w) > ϵ

        (
            w[mask], indices[mask]
        )
    end

    """
    Obtain IDW interpolation weights
    """
    function IDW_weights(
        X::AbstractMatrix, 
        indices::AbstractVector,
        x::AbstractVector
    )
        ϵ = eps(eltype(X))

        dX = X[:, indices] .- x
        
        distances = sum(
            dX .^ 2; dims = 1
        ) |> vec |> x -> sqrt.(x) .+ ϵ

        w = 1.0 ./ distances
        w ./= sum(w)

        mask = @. abs(w) > sqrt(ϵ)

        (
            w[mask], indices[mask]
        )
    end

    """
    $TYPEDSIGNATURES

    Obtain interpolator struct.

    Uses first index of an array for point indexing if `first_index = true`
        (def. false).

    Uses `k` closest points as stencils (def. `2^ndims`).
    """
    function Interpolator(
        X::AbstractMatrix, Xc::AbstractMatrix,
        tree::Union{KDTree, Nothing} = nothing;
        first_index::Bool = false,
        linear::Bool = true,
        k::Int = 0,
    )
        if first_index
            X = permutedims(X)
            Xc = permutedims(Xc)
        end

        if k == 0
            k = 2 ^ size(X, 1)
        end

        if isnothing(tree)
            tree = KDTree(X)
        end

        get_weights = (
            linear ?
            (X, idxs, x) -> linear_weights(X, idxs, x) :
            (X, idxs, x) -> IDW_weights(X, idxs, x)
        )

        idxs, ws = let tups = map(
            x -> let idxs = knn(
                tree, x, k
            )[1]
                get_weights(X, idxs, x)
            end,
            eachcol(Xc)
        )
            (
                map(t -> t[2], tups),
                map(t -> t[1], tups),
            )
        end

        Accumulator(
            idxs, ws;
            first_index = first_index
        )
    end

    """
    $TYPEDSIGNATURES

    Get domain for one or more interpolators.
    Returns a vector of domain indices and a dictionary 
    mapping previous indexes to indices in the domain.
    """
    function domain(
        intps::Accumulator...
    )
        idxs = let idxs = Set{Int64}()
            for intp in intps
                for (_, stencil, _) in values(intp.stencils)
                    for i in stencil
                        push!(idxs, i)
                    end
                end
            end

            idxs |> collect |> sort
        end

        (
            idxs,
            Dict(
                [k => i for (i, k) in enumerate(idxs)]
            )
        )
    end

    """
    $TYPEDSIGNATURES

    Re-index interpolator to handle new domain
    """
    function re_index!(
        intp::Accumulator, hmap::Dict{Int64, Int64}
    )
        for (_, stencil, _) in values(intp.stencils)
            stencil .= map(
                i -> hmap[i], stencil
            )
        end
    end

    """
    $TYPEDSIGNATURES

    Obtain coarsener and prolongator operators for the n-th grid level.
    The original grid points are re-sampled by selecting every `2^N`-th point,
    if `N` is the spatial dimensionality.

    Optionally, a cell volume for each grid point may be provided.
    """
    function coarsener_and_prolongator(
        X::AbstractMatrix, n::Int64, volumes::Union{AbstractVector, Nothing} = nothing;
        first_index::Bool = false, 
        linear::Bool = false,
        k::Int = 0,
    )
        X_for_tree = X
        if first_index
            X_for_tree = permutedims(X_for_tree)
        end

        if isnothing(volumes)
            volumes = similar(X_for_tree, (size(X_for_tree, 2),))
            volumes .= 1.0
        end

        N = size(X_for_tree, 1)
        Xc = selectdim(
            X,
            (first_index ? 1 : 2),
            1:(2 ^ (N * n)):size(X_for_tree, 2)
        )

        Xc_for_tree = Xc
        if first_index
            Xc_for_tree = permutedims(Xc_for_tree)
        end

        tree_coarse = KDTree(Xc_for_tree)

        idxs, _ = nn(tree_coarse, X_for_tree)

        # build coarsener based on closest clusters
        stencils = [
            Int64[] for i = 1:size(Xc_for_tree, 2)
        ]
        for (k, i) in enumerate(idxs)
            push!(stencils[i], k)
        end

        weights = [
            let v = @view volumes[i]
                v ./ sum(v)
            end for i in stencils
        ]

        coarsener = Accumulator(stencils, weights;
            first_index = first_index)

        prolongator = Interpolator(
            Xc, X, tree_coarse;
            first_index = first_index, linear = linear, k = k,
        )

        (coarsener, prolongator)
    end

    """
    $TYPEDSIGNATURES

    Function to perform under-relaxation upon a given grid, given coarsener and prolongator
    from the finest grid. If not provided, the system is under-relaxed at the current level.

    Returns proposed corrections to `v` and the corresponding variations to `r`.
    """
    function underrelax(
        J, r::AbstractArray,
        coarsener = identity, prolongator = identity
    )
        e = eps(eltype(r))
        v = r |> coarsener |> prolongator

        Jv = J(v)
        Jvc = coarsener(Jv)
        rc = coarsener(r)

        a = (
            dot(Jvc, rc) / (
                dot(Jvc, Jvc) + e
            )
        )
        (
            - v .* a,
            - Jv .* a,
        )
    end

    """
    $TYPEDFIELDS

    Struct holding multigrid levels.
    """
    struct Multigrid
        coarseners::AbstractVector{Accumulator}
        prolongators::AbstractVector{Accumulator}
    end

    """
    $TYPEDSIGNATURES

    Constructor for a multigrid struct with `n_levels` levels. 

    The original grid points are re-sampled by selecting every `2^N`-th point,
    if `N` is the spatial dimensionality.

    Optionally, a cell volume for each grid point may be provided.
    """
    function Multigrid(
        X::AbstractMatrix, n_levels::Int64, volumes::Union{AbstractVector, Nothing} = nothing;
        first_index::Bool = false, 
        linear::Bool = false,
        k::Int = 0,
    )
        coarseners = Accumulator[]
        prolongators = Accumulator[]

        for n = 1:n_levels
            c, p = coarsener_and_prolongator(X, n, volumes;
                first_index = first_index, linear = linear, k = k)

            push!(coarseners, c)
            push!(prolongators, p)
        end

        Multigrid(coarseners, prolongators)
    end

    """
    $TYPEDSIGNATURES

    Run `n_cycles` M-cycles on multigrid levels to solve Newton-Rhapson system.

    Uses Jacobian-free finite difference step `h`. Returns correction to `x`, 
    final linear system residuals and residual norm reduction factor.
    """
    function (mgrid::Multigrid)(
        f, x::AbstractArray; 
        n_cycles::Int = 1000,
        rtol::Real = 0.01, atol::Real = 1e-7,
        h::Real = 1e-6,
    )
        n_levels = mgrid.coarseners |> length

        r0 = f(x)
        r = copy(r0)

        s = similar(r)
        s .= 0.0

        J = v -> (f(x .+ v .* h) .- r0) ./ h

        nr0 = norm(r0)
        for _ = 1:n_cycles
            nr = norm(r)
            if nr < max(atol, rtol * nr0)
                break
            end

            ds, dr = underrelax(J, r)
            s .+= ds
            r .+= dr
            
            for i = [
                1:n_levels; (n_levels-1):-1:1
            ]
                ds, dr = underrelax(J, r, mgrid.coarseners[i], mgrid.prolongators[i])
                s .+= ds
                r .+= dr
            end

            ds, dr = underrelax(J, r)
            s .+= ds
            r .+= dr
        end
        rf = norm(r) / (nr0 + eps(eltype(r0)))

        s, r, rf
    end

end
