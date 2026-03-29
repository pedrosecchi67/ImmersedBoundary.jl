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
        i1 = ones(Int64, length(q))
        i2 = fill(length(u), length(q))
        complete = falses(length(q))

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
        κ::Float64
        A::Float64
        β::Float64
        βstar::Float64
        D::Float64
        A⁺::Float64
    end

    """
    $TYPEDSIGNATURES

    Constructor for a wall function.

    Uses Wilcox's expressions for near-wall `k⁺`,
    and Nezu and Nakagawa's expressions for it in the log-law region.
    Van Driest's approximation for `μ⁺` is used via differential equations
    to calculate an interpolated solution for `y⁺`.
    """
    function WallFunction(
        ; 
        h0::Float64 = 0.01, 
        growth_ratio::Float64 = 1.05,
        y⁺_max::Float64 = 30000.0,
        constant_layer_y⁺::Float64 = 15.0,
        κ::Float64 = 0.41, A::Float64 = 19.0,
        β::Float64 = 0.075, βstar::Float64 = 0.09,
        D::Float64 = 4.2, A⁺::Float64 = 360.0,
    )
        h = h0

        ϵ = eps(Float64) |> sqrt
        yps = [ϵ, h]
        ups = [ϵ, h]

        while yps[end] < y⁺_max
            yp = yps[end]
            μ⁺ = κ * yp * (1.0 - exp(- yp / A)) ^ 2
            du!dy = 1.0 / (μ⁺ + 1.0)

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
        ϵ = eps(Float64)
        Rey = @. clamp(abs(Rey), ϵ, Inf64)

        # interpolated from previous diff. equation solution
        y⁺ = 10.0 .^ linear_interpolation(
            log10.(Rey), wf.log10Rey, wf.log10y⁺
        )
        
        u⁺ = Rey ./ y⁺

        # from van Driest
        μ⁺ = @. wf.κ * y⁺ * (1.0 - exp(- y⁺ / wf.A)) ^ 2
        du⁺!dy⁺ = @. 1.0 / (1.0 + μ⁺)

        # from Nakagawa-Nezu
        k⁺ = @. min(
            y⁺ ^ 2 / (6.0 * wf.βstar / wf.β - 2.0),
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

    export Smagorinsky_νSGS

    """
    $TYPEDSIGNATURES

    Obtain `νSGS` as per the Smagorinsky turbulence model.
    
    Uses `νSGS = (CₛΔ)²|S|` for 3D, or `νSGS = (CₛΔ)²|ω|` for 2D.

    `velocity_gradient` is a matrix such that `velocity_gradient[i, j]`
    indicates the gradient of vel. component `i` along dimension `j`.
    """
    function Smagorinsky_νSGS(
        Δ::AbstractVector, velocity_gradient::AbstractMatrix;
        Cₛ::Real = 0.17,
    )
        if size(velocity_gradient, 1) == 2 # 2D
            ω = (
                velocity_gradient[2, 1] .- velocity_gradient[1, 2]
            )

            return (
                @. (Cₛ * Δ) ^ 2 * abs(ω)
            )
        end

        SijSij = similar(Δ)
        SijSij .= 0.0
        for i = 1:size(velocity_gradient, 1)
            for j = 1:size(velocity_gradient, 2)
                SijSij .+= (
                    (velocity_gradient[i, j] .+ velocity_gradient[j, i]) ./ 2
                ) .^ 2
            end
        end

        @. (Cₛ * Δ) ^ 2 * sqrt(2 * SijSij)
    end

end
