module GeometricMultigrid

    using DocStringExtensions

    using LinearAlgebra
    using NearestNeighbors

    using Random: randperm

    include("accumulator.jl")
    using .ArrayAccumulator

    export Multigrid, backward_euler

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
        random_permutation::Bool = false,
    )
        X_for_tree = X
        if first_index
            X_for_tree = permutedims(X_for_tree)
        end

        if isnothing(volumes)
            volumes = similar(X_for_tree, (size(X_for_tree, 2),))
            volumes .= 1
        end

        N = size(X_for_tree, 1)
        Xc = let _X = X
            if random_permutation
                if first_index
                    perm = randperm(size(_X, 1))
                    _X = _X[perm, :]
                else
                    perm = randperm(size(_X, 2))
                    _X = _X[:, perm]
                end
            end

            selectdim(
                _X,
                (first_index ? 1 : 2),
                1:(2 ^ (N * n)):size(X_for_tree, 2)
            )
        end

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

        # build prolongator by inverting coarsener
        pstencils = [
            Int64[0] for i = 1:size(X_for_tree, 2)
        ]
        for (k, idxs) in enumerate(stencils)
            for i in idxs
                pstencils[i][1] = k
            end
        end

        prolongator = Accumulator(pstencils; first_index = first_index)

        (coarsener, prolongator)
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

    If `random_permutation` is true, the points are randomly permuted
    to reduce the likelyhood of accumulations.

    If `first_index` is true (default), the first index of an array refers to 
    the grid point. Otherwise, if false, the last index of an array refers to the
    grid point.
    """
    function Multigrid(
        X::AbstractMatrix, n_levels::Int64, volumes::Union{AbstractVector, Nothing} = nothing;
        first_index::Bool = true, 
        random_permutation::Bool = false,
    )
        coarseners = Accumulator[]
        prolongators = Accumulator[]

        for n = 1:n_levels
            c, p = coarsener_and_prolongator(X, n, volumes;
                first_index = first_index,
                random_permutation = random_permutation)

            push!(coarseners, c)
            push!(prolongators, p)
        end

        Multigrid(coarseners, prolongators)
    end

end
