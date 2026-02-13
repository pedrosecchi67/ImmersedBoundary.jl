module Turbulence

    using ..LinearAlgebra
    using ..DocStringExtensions

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
        ω::Real = 0.5,
    )
        Rey = @. y * u / ν
        ypmin, ypmax = range

        y⁺ = nothing
        if u isa AbstractArray
            y⁺ = similar(u)
            y⁺ .= ypmax

            for _ = 1:n_iter
                @. y⁺ = y⁺ + (
                    clamp(
                        Rey / law(y⁺), ypmin, ypmax
                    ) - y⁺
                ) * ω
            end
        else
            y⁺ = ypmax

            for _ = 1:n_iter
                y⁺ = y⁺ + (
                    clamp(
                        Rey / law(y⁺), ypmin, ypmax
                    ) - y⁺
                ) * ω
            end
        end

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