module NNInterpolator

    using DocStringExtensions

    using LinearAlgebra
    using NearestNeighbors

    export Interpolator, KDTree

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

end
