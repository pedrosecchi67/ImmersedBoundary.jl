module Turbulence

    using DocStringExtensions

    """
    Function for a vectorized binary search.

    Looks for the index of the last element in `u`
    which is smaller than an entry in vector of query points `q`
    """
    function vector_binary_search(
        q::AbstractVector, u::AbstractVector
    )
        i1 = similar(q, Int32, length(q))
        i2 = similar(q, Int32, length(q))
        i1 .= 1
        i2 .= length(u)

        complete = similar(q, Bool, length(q))
        complete .= false

        im = similar(i1)
        while !all(complete)
            @. im = (i1 + i2) ÷ 2
            um = view(u, im)

            @. i1 = (um < q) * im + (um >= q) * i1
            @. i2 = (um < q) * i2 + (um >= q) * im

            @. complete = complete || (i2 <= i1 + 1)
        end

        i1
    end

    """
    Linear interpolation
    """
    function linear_interpolation(
        xquery::AbstractVector, x::AbstractVector, y::AbstractVector
    )
        i = vector_binary_search(xquery, x)
        inxt = i .+ 1

        xl = @view x[i]
        xn = @view x[inxt]
        yl = @view y[i]
        yn = @view y[inxt]

        @. yl + (yn - yl) * (xquery - xl) / (xn - xl)
    end

    export WallFunction

    """
    $TYPEDFIELDS

    Struct to hold wall function definitions
    """
    struct WallFunction
        log10Rey::AbstractVector
        log10y⁺::AbstractVector
        κ::Real
        A::Real
        β::Real
        βstar::Real
        D::Real
        A⁺::Real
    end

    """
    $TYPEDSIGNATURES

    Constructor for a wall function.

    Uses Wilcox's expressions for near-wall `k⁺`,
    and Nezu and Nakagawa's expressions for it in the log-law region.
    Van Driest's approximation for `μ⁺` is used via differential equations
    to calculate an interpolated solution for `y⁺`.

    Wake function `g` is calculated as per a squared-cosinoidal wake layer profile.
    """
    function WallFunction(
        ; 
        h0::Real = 0.01f0, 
        growth_ratio::Real = 1.05f0,
        y⁺_max::Real = 10000.0f0,
        constant_layer_y⁺::Real = 15.0f0,
        κ::Real = 0.41f0, A::Real = 19.0f0,
        β::Real = 0.075f0, βstar::Real = 0.09f0,
        D::Real = 4.2f0, A⁺::Real = 360.0f0,
        Cf_max::Real = 0.01f0,
    )
        h = h0

        ϵ = 1f-10
        yps = [ϵ, h]
        ups = [ϵ, h]

        while yps[end] < y⁺_max
            yp = yps[end]
            μ⁺ = κ * yp * (1.0f0 - exp(- yp / A)) ^ 2
            du!dy = 1.0 / (μ⁺ + 1.0f0)

            push!(
                yps, yp + h
            )
            push!(
                ups, ups[end] + h * du!dy
            )

            if yp > constant_layer_y⁺
                h *= growth_ratio
            end
        end

        g = @. sin(
            yps / y⁺_max * π / 2
        ) ^ 2
        upmax = ups[end] + sqrt(2.0f0 / Cf_max)

        @. ups = g * upmax + (1.0f0 - g) * ups

        Rey = @. ups * yps

        WallFunction(
            log10.(Rey),
            log10.(yps),
            κ, A, β, βstar, D, A⁺
        )
    end

    """
    $TYPEDSIGNATURES

    Obtain named tuple with `y⁺, u⁺, μ⁺, k⁺, du⁺!dy⁺` given a set of `Rey` values 
    (Reynolds number in respect to local, laminar viscosity and the `y` 
    position of the first cell center).
    """
    function (wf::WallFunction)(Rey::AbstractVector)
        ϵ = eps(eltype(Rey))
        Rey = @. clamp(abs(Rey), ϵ, Inf32)

        # interpolated from previous diff. equation solution
        log10Rey = log10.(Rey)
        y⁺ = 10.0f0 .^ linear_interpolation(
            log10Rey, wf.log10Rey, wf.log10y⁺
        )
        
        u⁺ = Rey ./ y⁺

        # from van Driest
        μ⁺ = @. wf.κ * y⁺ * (1.0f0 - exp(- y⁺ / wf.A)) ^ 2
        du⁺!dy⁺ = @. 1.0f0 / (1.0f0 + μ⁺)

        # from Nakagawa-Nezu
        k⁺ = @. min(
            y⁺ ^ 2 / (6.0f0 * wf.βstar / wf.β - 2.0f0),
            wf.D * exp(
                - y⁺ / wf.A⁺
            )
        )

        (
            y⁺ = y⁺, 
            u⁺ = u⁺, 
            μ⁺ = μ⁺,
            k⁺ = k⁺,
            du⁺!dy⁺ = du⁺!dy⁺,
        )
    end

    """
    $TYPEDSIGNATURES

    Obtain named tuple with `uτ, νₜ, k, ω, ϵ, du!dn` given a set of `y, u, ν` values 
    """
    function (wf::WallFunction)(
        y::AbstractVector, u::AbstractVector, ν::AbstractVector
    )
        nt = wf(u .* y ./ ν)

        uτ = u ./ nt.u⁺

        νₜ = nt.μ⁺ .* ν
        k = nt.k⁺ .* uτ .^ 2
        ω = k ./ νₜ
        ϵ = @. wf.βstar * ω * k

        du!dn = @. nt.du⁺!dy⁺ * uτ ^ 2 / ν

        (
            uτ = uτ,
            νₜ = νₜ,
            k = k,
            ω = ω,
            ϵ = ϵ,
            du!dn = du!dn,
        )
    end

    export shear_rate

    """
    $TYPEDSIGNATURES

    Obtain `sqrt(2 * SijSij)`.

    `velocity_gradient` is a matrix such that `velocity_gradient[i, j]`
    indicates the gradient of vel. component `i` along dimension `j`.
    """
    function shear_rate(
        velocity_gradient::AbstractMatrix
    )
        SijSij = similar(velocity_gradient[1, 1])
        SijSij .= 0
        for i = 1:size(velocity_gradient, 1)
            for j = 1:size(velocity_gradient, 2)
                SijSij .+= (
                    (velocity_gradient[i, j] .+ velocity_gradient[j, i]) ./ 2
                ) .^ 2
            end
        end

        @. sqrt(2 * SijSij)
    end

    export Smagorinsky_νSGS

    """
    $TYPEDSIGNATURES

    Obtain `νSGS` as per the Smagorinsky turbulence model.
    `S` is the norm of vorticity, `sqrt(2 * SijSij)`
    """
    Smagorinsky_νSGS(
        Δ::AbstractVector, S::AbstractVector;
        Cₛ::Real = 0.17f0,
    ) = (@. (Cₛ * Δ) ^ 2 * S)

    export Wray_Argawal

    """
    $TYPEDSIGNATURES

    Obtain closure for a 'simplified' Wray-Argawal turbulence model
    which collapses all constants to the `k-ω` values.

    Remember: BCs involve using `R∞ = 3ν` and `R=0` at walls!

    The return value is a tuple with entries:

    ```
    (
        νₜ = R, # just to make sure you know ;)
        νR = (dissipation rate for R),
        S = (source term)
    )
    ```

    Such that:

    ```
    Rₜ = - ∇⋅(uR) + ∇⋅[(ν + νR) ∇R] + S
    ```
    """
    function Wray_Argawal(
        R::AbstractVector, S::AbstractVector,
        ∇R::AbstractMatrix, ∇S::AbstractMatrix;
        σR::Real = 0.72f0, C₁::Real = 0.0829f0, κ::Real = 0.41f0,
    )
        ϵ = eps(eltype(R))

        C₂ = σR + C₁ / κ ^ 2

        S = let ∇R∇S = sum(∇R .* ∇S; dims = 2) |> vec
            @. C₁ * R * S + C₂ * ∇R∇S * (R / (S + ϵ))
        end

        (
            νₜ = R,
            νR = R .* σR,
            S = S
        )
    end

end
