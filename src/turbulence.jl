module Turbulence

    using ..LinearAlgebra
    using ..DocStringExtensions

    """
    Secant method implementation
    """
    function secant(f, x1, x2, n_iter::Int = 20)
        f1 = f(x1)
        f2 = f(x2)
        h = eps(eltype(f1))

        xsolve = @. (
            abs(f1) * x2 + abs(f2) * x1
        ) / (abs(f1) + abs(f2) + h)
        fsolve = f(xsolve)

        for nit = 1:n_iter
            if x1 isa AbstractArray
                isfirst = @. fsolve * f2 < fsolve * f1

                @. x1 = x1 * (1.0 - isfirst) + xsolve * isfirst
                @. x2 = x2 * isfirst + xsolve * (1.0 - isfirst)

                xsolve = @. (
                    abs(f1) * x2 + abs(f2) * x1
                ) / (abs(f1) + abs(f2) + h)
                fsolve .= f(xsolve)
            else
                isfirst = fsolve * f2 < fsolve * f1

                x1 = x1 * (1.0 - isfirst) + xsolve * isfirst
                x2 = x2 * isfirst + xsolve * (1.0 - isfirst)

                xsolve = (
                    abs(f1) * x2 + abs(f2) * x1
                ) / (abs(f1) + abs(f2) + h)
                fsolve = f(xsolve)
            end
        end

        xsolve
    end

    """
    $TYPEDSIGNATURES

    Obtain `y⁺` given a law of the wall (`u⁺(y⁺)`),
    local dynamic viscosity at the wall, a wall distance and an
    external velocity. Also returns u⁺, uτ and νt
    """
    function resolve_LOTW(
        law, ν, y, u; 
        range::Tuple = (0.2, 300.0),
        n_iter::Int = 20,
    )
        f = y⁺ -> let u⁺ = law.(y⁺)
            uτ = @. u / u⁺
            @. uτ * y / ν - y⁺
        end

        x1 = similar(u)
        x1 .= range[1]
        x2 = similar(u)
        x2 .= range[2]

        y⁺ = secant(
            f, x1, x2, n_iter
        )
        u⁺ = law.(y⁺)
        uτ = @. u / u⁺

        νt = @. (
            0.41 * uτ * y * (1.0 - exp(- y⁺ / 19.0)) ^ 2
        )

        (y⁺, u⁺, uτ, νt)
    end

    """
    $TYPEDSIGNATURES

    Von-Karman Law of the Wall
    """
    function Von_Karman(y⁺::Real)
        if y⁺ < 1.0
            return y⁺
        end

        min(
            5.6 * log10(y⁺) + 4.9, y⁺
        )
    end

    """
    $TYPEDSIGNATURES

    Obtain eddy viscosity as per Smagorinsky's turbulence model.
    Receives grid size `Δ` as an input.

    The velocity gradients should be specified in a matrix of arrays,
    such that `velocity_gradient[i, j]` indicates the gradient of vel.
    component `j` along dimension `i`.
    """
    function Smagorinsky_νₜ(
        Δ::AbstractArray, velocity_gradient::AbstractMatrix;
        Cₛ::Real = 0.16,
    )
        SijSij = first(velocity_gradient) |> similar
        SijSij .= 0.0

        for i = 1:size(velocity_gradient, 1)
            for j = 1:size(velocity_gradient, 2)
                SijSij .+= (
                    (
                        velocity_gradient[i, j] .+ velocity_gradient[j, i]
                    ) ./ 2
                ) .^ 2
            end
        end

        @. Cₛ * Δ ^ 2 * sqrt(2 * SijSij)
    end

end # module Turbulence