module IBL

    export θ_closure, m_closure

    using DocStringExtensions

    """
    Module for White's method correlation functions
    """
    module White

        function H(
            Λ, Reθ
        )

            _rth = @. clamp(Reθ, 1e2, 1e6)
            _Λ = @. clamp(Λ, -4.52859e-3, 4.5e-3)
            _L = @. log10(_rth)

            H = @. - 4.072 * log(_Λ + 4.5286e-3) / (- 0.1331 * _L ^ 2 + 1.3061 * _L + 6.0) - 1.085
            @. clamp(H, 1.0, 2.38)

        end

        function Cf(Λ, Reθ)

            _rth = @. clamp(Reθ, 1e2, 1e6)
            _L = @. log10(_rth)

            _H = @. H(Λ, Reθ)

            Cf = @. 0.3 * exp(- 1.33 * _H) / (
                _L ^ (1.74 + 0.31 * _H)
            )

        end

    end
    using .White

    """
    $TYPEDSIGNATURES

    Obtain named tuple with boundary layer parameters from the local kin. energy thickness.
    `velocity` is a matrix with each row referring to a mesh cell/point,
    and `pressure_gradient` is, similarly, a matrix with each row referring
    to a mesh cell/point. It defaults to 0.

    Density at each control point may be specified in argument `ρ`.

    For IBL equations:

    ```
    ṁ = - ∇ ⋅ (uj) + τ

    m = ρVθH
    j = ρVθ
    ```

    The output includes fields `θ, Cf, H, δstar, V, dV!ds, ρ, m, j, τ`.
    """
    function θ_closure(
        θ::AbstractVector,
        velocity::AbstractMatrix, 
        ν::Union{Float64, AbstractVector};
        pressure_gradient::Union{Float64, AbstractMatrix} = 0.0,
        ρ::Union{Float64, AbstractVector} = 1.0,
    )
        ϵ = sqrt(eps(Float64))

        u = velocity
        V = sum(u .^ 2; dims = 2) |> vec |> x -> sqrt.(x) .+ ϵ
        pₓ = sum(
            u .* pressure_gradient ./ V; dims = 2
        ) |> vec

        dV!ds = @. pₓ / (V * ρ)
        Λ = @. dV!ds * θ / V
        Reθ = @. θ * V / ν

        Cf = White.Cf(Λ, Reθ)
        H = White.H(Λ, Reθ)

        δstar = H .* θ

        (
            θ = copy(θ),
            Cf = Cf, H = H,
            δstar = δstar,
            V = V,
            dV!ds = dV!ds,
            ρ = copy(ρ),
            m = δstar .* V .* ρ,
            j = θ .* V .* ρ,
            τ = Cf .* V .^ 2 .* ρ ./ 2,
        )
    end

    """
    $TYPEDSIGNATURES

    Obtain named tuple with boundary layer parameters from the local velocity deffect.
    `velocity` is a matrix with each row referring to a mesh cell/point,
    and `pressure_gradient` is, similarly, a matrix with each row referring
    to a mesh cell/point. It defaults to 0.

    Density at each control point may be specified in argument `ρ`.

    For IBL equations:

    ```
    ṁ = - ∇ ⋅ (uj) + τ

    m = ρVθH
    j = ρVθ
    ```

    The output includes fields `θ, Cf, H, δstar, V, dV!ds, ρ, m, j, τ`.

    Parameters `n_iter` and `ω` are used for fixed-point iterations for 
    shape parameter.
    """
    function m_closure(
        m::AbstractVector,
        velocity::AbstractMatrix, 
        ν::Union{Float64, AbstractVector};
        pressure_gradient::Union{Float64, AbstractMatrix} = 0.0,
        ρ::Union{Float64, AbstractVector} = 1.0,
        n_iter::Int = 20, ω::Real = 0.8,
    )
        ϵ = sqrt(eps(Float64))

        u = velocity
        V = sum(u .^ 2; dims = 2) |> vec |> x -> sqrt.(x) .+ ϵ
        pₓ = sum(
            u .* pressure_gradient ./ V; dims = 2
        ) |> vec

        dV!ds = @. pₓ / (V * ρ)

        Λ_multiplier = @. dV!ds / V ^ 2 * ρ
        Reθ_multiplier = @. 1.0 / ν / ρ

        H = similar(m)
        H .= 1.5

        Λ = Λ_multiplier .* m ./ H
        Reθ = Reθ_multiplier .* m ./ H

        Cf = White.Cf(Λ, Reθ)
        H = White.H(Λ, Reθ)

        for _ = 1:n_iter
            Λ .= Λ_multiplier .* m ./ H
            Reθ .= Reθ_multiplier .* m ./ H

            Cf .= White.Cf(Λ, Reθ) .* ω .+ Cf .* (1.0 - ω)
            H .= White.H(Λ, Reθ) .* ω .+ H .* (1.0 - ω)
        end

        δstar = @. m / ρ / V
        θ = δstar ./ H

        (
            θ = θ,
            Cf = Cf, H = H,
            δstar = δstar,
            V = V,
            dV!ds = dV!ds,
            ρ = copy(ρ),
            m = copy(m),
            j = θ .* V .* ρ,
            τ = Cf .* V .^ 2 .* ρ ./ 2,
        )
    end

end
