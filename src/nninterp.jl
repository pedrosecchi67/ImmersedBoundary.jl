module NNInterpolator

    using DocStringExtensions

    using LinearAlgebra
    using NearestNeighbors

    export Interpolator, KDTree

    include("accumulator.jl")
    using .ArrayAccumulator

    """
    Obtain linear interpolation weights
    """
    function linear_weights(
        X::AbstractMatrix, 
        indices::AbstractVector,
        x::AbstractVector
    )
        Tf = eltype(X)
        ϵ = eps(Tf)

        dX = X[:, indices] .- x
        
        distances = sum(
            dX .^ 2; dims = 1
        ) |> vec |> x -> sqrt.(x) .+ ϵ

        w = Tf(1.0) ./ distances
        w = let A = [
            dX' ones(Tf, size(dX, 2))
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
        Tf = eltype(X)
        ϵ = eps(Tf)

        dX = X[:, indices] .- x
        
        distances = sum(
            dX .^ 2; dims = 1
        ) |> vec |> x -> sqrt.(x) .+ ϵ

        w = Tf(1.0) ./ distances
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
