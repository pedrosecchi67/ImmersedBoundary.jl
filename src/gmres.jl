module GMRES

    using DocStringExtensions

    using LinearAlgebra

    export Linearization, gmres

    """
    $TYPEDFIELDS

    Struct describing a linearized function
    """
    struct Linearization
        f
        x0::AbstractArray
        f0::AbstractArray
        h::Float64
    end

    """
    $TYPEDSIGNATURES

    Construct linearization.
    If not provided, `h = 10 * (eltype(x0) |> eps |> sqrt)`.

    Also returns RHS of Newton system
    """
    function Linearization(
        f,
        x0::AbstractArray;
        h::Real = 0.0
    )
        x0 = copy(x0)

        if h == 0.0
            h = 10 * (
                eltype(x0) |> eps |> sqrt
            )
        end

        f0 = f(x0)

        return (
            Linearization(
                f, x0, f0, h
            ), - f0
        )
    end

    """
    $TYPEDSIGNATURES

    Evaluate linearization
    """
    function (lin::Linearization)(
        v::AbstractArray
    )
        (
            lin.f(
                (@. lin.x0 + lin.h * v)
            ) .- lin.f0
        ) ./ lin.h
    end

    """
    $TYPEDSIGNATURES

    Solve GMRES linear problem with low memory footprint
    """
    function solve_linear_problem(
        AV::AbstractVector,
        b::AbstractArray
    )
        N = length(AV)
        T = eltype(b)

        H = Matrix{T}(undef, N, N)
        y = Vector{T}(undef, N)

        for i = 1:N
            y[i] = (b ⋅ AV[i])

            for j = 1:N
                if i > j
                    H[i, j] = H[j, i]
                else
                    H[i, j] = (AV[i] ⋅ AV[j])
                end
            end
        end

        pinv(H) * y
    end

    """
    $TYPEDSIGNATURES

    Obtain corrections `Δx` after a few iterations of GMRES
    for system `Ax = b`.
    Operator `A` is described by an array-valued function so that `A(v) = A * v`.

    Also returns remaining residual at the end of the iteration.
    """
    function gmres(
        A, b::AbstractArray, n_iter::Int64;
        preconditioner = x -> x,
    )
        ϵ = eltype(b) |> eps

        V = []
        AV = []

        arnoldi! = v -> begin
            v = copy(v)

            for vv in V
                proj = v ⋅ vv
                @. v -= proj * vv
            end

            v ./= (norm(v) + ϵ)
            Av = A(v)

            push!(V, v)
            push!(AV, Av)
            return
        end

        for nit = 1:n_iter
            arnoldi!(
                preconditioner(
                    nit == 1 ?
                    b :
                    AV[end]
                )
            )
        end

        αs = solve_linear_problem(AV, b)

        # re-using vectors to save memory
        s = V[1]
        r = AV[1]
        s .*= αs[1]
        r .*= (- αs[1])
        @. r += b

        for (α, v, Av) in zip(αs[2:end], V[2:end], AV[2:end])
            @. s += α * v
            @. r -= α * Av
        end

        (s, r)
    end

end