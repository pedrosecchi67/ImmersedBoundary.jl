module PointImplicit

    export linearize, solve, Multigrid

    using LinearAlgebra

    using DocStringExtensions

    include("mgrid.jl")
    using .GeometricMultigrid

    """
    $TYPEDSIGNATURES

    Estimate diagonal terms of the Jacobian of a 
    function returning a vector or tensor.
    """
    function hutchinson_trick(
        f, x::AbstractVector, n_samples::Int;
        h::Real = 1e-6,
        pre_evaluated_fx = nothing
    )
        fx = (
            isnothing(pre_evaluated_fx) ?
            f(x) : pre_evaluated_fx
        )

        s = similar(fx)
        s .= 0
        xbuff = similar(x)

        J = v -> begin
            @. xbuff = x + v * h

            (f(xbuff) .- fx) ./ h
        end

        for i = 1:n_samples
            z = rand(
                Int32[-1, 1], length(x)
            )

            s .+= z .* J(z)
        end

        @. s / n_samples
    end

    """
    $TYPEDSIGNATURES

    Calculate diagonal blocks of a function taking/returning a matrix.
    Takes formats:

    ```
    Db = hutchinson_trick(
        f, X, 30
    )

    size(X) # (npoints, nvars)
    size(f(X)) # (npoints, nvars)
    size(Db) # (npoints, nvars, nvars)
    ```
    """
    function hutchinson_trick(
        f, X::AbstractMatrix, n_samples::Int;
        h::Real = 1e-6,
        pre_evaluated_fx = nothing,
    )
        Xbuff = copy(X)
        fX = (
            isnothing(pre_evaluated_fx) ? f(X) : pre_evaluated_fx
        )

        stack(
            i -> begin
                fv = x -> let Xbv = @view Xbuff[:, i]
                    Xbv .= x
                    fxb = f(Xbuff)

                    Xbuff[:, i] .= X[:, i]

                    fxb
                end

                hutchinson_trick(fv, X[:, i], n_samples; h =  h,
                    pre_evaluated_fx = fX)
            end,
            1:size(X, 2)
        )
    end

    """
    $TYPEDFIELDS

    Struct to define a linearization
    """
    struct Linearization
        f
        x::AbstractArray
        fx::AbstractArray
        h::Real
    end

    """
    $TYPEDSIGNATURES

    Evaluate Jacobian-vector-product given linearization
    """
    (lin::Linearization)(
        v::AbstractArray
    ) = (
        lin.f(lin.x .+ v .* lin.h) .- lin.fx
    ) ./ lin.h

    """
    $TYPEDFIELDS

    Struct to define a [block-]diagonal preconditioner.
    """
    struct PIPreconditioner{Tf, N}
        inverse_diagonal::AbstractArray{Tf, N}
    end

    function _inverse_blocks!(D::AbstractVector{Tf}) where {Tf <: AbstractFloat}
        D .= Tf(1.0) ./ (eps(Tf) .+ D)
    end
    function _inverse_blocks!(D::AbstractArray{Tf}) where {Tf <: AbstractFloat}
        # for each block...
        for J = eachslice(D; dims = 1)
            J .= pinv(J) # invert!
        end

        D
    end

    """
    $TYPEDSIGNATURES

    Obtain preconditioner-vector product
    """
    (
        prec::PIPreconditioner{Tf, 1}
    )(v::AbstractVector) where Tf = (
        v .* prec.inverse_diagonal
    )

    """
    $TYPEDSIGNATURES

    Obtain preconditioner-vector product
    """
    (
        prec::PIPreconditioner{Tf, N}
    )(v::AbstractMatrix) where {Tf, N} = let vr = reshape(v, size(v, 1), 1, :)
        sum(
            vr .* prec.inverse_diagonal; dims = 3
        ) |> x -> dropdims(
            x; dims = 3
        )
    end

    """
    $TYPEDSIGNATURES

    Linearize a function evaluation for a Newton-Rhapson system.

    Returns:

    ```
    A, b, D = linearize(f, x)

    # b = - f(x), residual
    # for Newton system Ax = b

    # D the [block-]diagonal preconditioner, 
    # a callable where D(b) ≈ x so that Ax = b.
    ```

    Note that `x` may be a matrix for a multi-variable system,
    with the first index identitying the control point, and the
    second, the variable.
    """
    function linearize(
        f, x::AbstractVecOrMat;
        n_hutchinson_samples::Int = 30,
        pre_evaluated_fx = nothing,
        h::Real = 1e-6,
    )
        fx = (
            isnothing(pre_evaluated_fx) ?
            f(x) : copy(pre_evaluated_fx)
        )
        x = copy(x)

        D = hutchinson_trick(
            f, x, n_hutchinson_samples;
            h = h, pre_evaluated_fx = pre_evaluated_fx
        )
        _inverse_blocks!(D)

        (
            Linearization(f, x, fx, h),
            - fx,
            PIPreconditioner(D)
        )
    end

    """
    $TYPEDSIGNATURES

    Obtain the coefficient `α` such that:

    ```
    α = argmin{|b - α × Av|}
    ```

    Also returns `Av`.
    """
    function proj_along(
        A::Linearization, v::AbstractVecOrMat, b::AbstractVecOrMat
    )
        ϵ = eps(eltype(v))
        Av = A(v)
        
        (
            (
                Av ⋅ b
            ) / (
                Av ⋅ Av + ϵ
            ), Av
        )
    end

    """
    $TYPEDSIGNATURES

    Solve linear system using block-implicit preconditioner
    and relaxation.

    Returns correction array and `norm(r)/norm(r0)` ratio.
    Default tolerances are `atol = 1e-7, rtol = 1e-3`.
    Runs `n_iter` iterations (def. 100).

    Accepts an optional, geometric multigrid object in argument
    `multigrid`, case in which V-cycles are ran.

    `n_inner` iterations are ran per multigrid level.
    """
    function solve(
        A::Linearization, b::AbstractVector,
        prec::PIPreconditioner;
        n_iter::Int = 100,
        n_inner::Int = 1,
        rtol::Real = 1e-2,
        atol::Real = 1e-7,
        multigrid = nothing,
        verbose::Bool = false,
    )
        ϵ = eps(eltype(b))

        nr0 = norm(b)
        nr = nr0

        x = similar(b)
        x .= 0

        r = copy(b)

        n_levels = (
            isnothing(multigrid) ?
            0 : length(multigrid.coarseners)
        )
        n_mgrid = n_levels

        verbose && println("Beginning point-implicit solution")
        verbose && println(
            "Iteration |r|/|r0|"
        )

        for nit = 1:n_iter
            for nin = 1:n_inner
                s = prec(r)

                if n_mgrid > 0
                    s .= (
                        s |> multigrid.coarseners[n_mgrid] |> multigrid.prolongators[n_mgrid]
                    )
                end

                α, As = proj_along(A, s, r)

                x .+= s .* α
                r .-= As .* α

                # second iteration: residual itself
                s .= r ./ (
                    ϵ + maximum(abs, r)
                )

                α, As = proj_along(A, s, r)

                x .+= s .* α
                r .-= As .* α

                nr = norm(r)

                verbose && println(
                    "$((nit - 1) * n_inner + nin)       $(nr / (nr0 + ϵ))"
                )

                if nr < nr0 * rtol + atol
                    return (
                        x, nr / (nr0 + ϵ)
                    )
                end
            end

            if n_mgrid == 0
                n_mgrid = n_levels
            else
                n_mgrid -= 1 # cycle from coarsest to finest
            end
        end

        return (
            x, nr / (nr0 + ϵ)
        )
    end

end
