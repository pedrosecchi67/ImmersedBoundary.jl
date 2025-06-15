module CFD

    using ..LinearAlgebra

    using ..DocStringExtensions

    """
    $TYPEDFIELDS

    Struct defining an ideal gas
    """
    struct Fluid
        R::Real
        γ::Real
    end

    """
    $TYPEDSIGNATURES

    Constructor for a fluid
    """
    Fluid(
        ;
        R::Real = 283.0,
        γ::Real = 1.4,
    ) = Fluid(R, γ)

    """
    $TYPEDSIGNATURES

    Convert state to primitive variables for a fluid.

    Receives `ρ, eₜ, ρu, ρv[, ρw]` in scalar or array format.

    Returns `p, T, u, v[, w]`.

    Example:

    ```
    p, T, u, v = state2primitive(fld, ρ, et, ρu, ρv)
    ```
    """
    function state2primitive(
        fld::Fluid,
        ρ, E, ρu...
    )

        u = map(ρui -> (ρui ./ ρ), ρu)

        vel = sqrt.(
            sum(
                ui -> ui .^ 2,
                u
            )
        )

        Cv = fld.R / (fld.γ - 1.0)
        T = @. (E - vel ^ 2 * ρ / 2) / ρ / Cv

        p = @. fld.R * ρ * T

        (p, T, u...)

    end

    """
    $TYPEDSIGNATURES

    Obtain speed of sound from temperature
    """
    speed_of_sound(fld::Fluid, T) = (
        @. sqrt(fld.γ * fld.R * clamp(T, 10.0, Inf64))
    )

    """
    $TYPEDSIGNATURES

    Obtain state variables from primitive variables.

    Receives `p, T, u, v[, w]` in scalar or array format.

    Returns `ρ, eₜ, ρu, ρv[, ρw]`.

    Example:

    ```
    ρ, et, ρu, ρv = primitive2state(fld, p, T, u, v)
    ``` 
    """
    function primitive2state(fld::Fluid, p, T, u...)

        ρ = @. p / (fld.R * T)

        vel = sqrt.(
            sum(
                ui -> ui .^ 2,
                u
            )
        )

        Cv = fld.R / (fld.γ - 1.0)
        et = @. ((Cv * T + vel ^ 2 / 2) * ρ)

        ρu = map(ui -> ρ .* ui, u)

        (ρ, et, ρu...)

    end

    """
    $TYPEDSIGNATURES

    Utility function to obtain RMS of residual arrays
    """
    rms(a::AbstractArray) = sqrt(
        sum(
            a .^ 2
        ) / length(a)
    )

    """
    Evaluate HLL flux
    """
    _hll(ul::Real, ur::Real, fl::Real, fr::Real, Sl::Real, Sr::Real) = (
        Sr * fl - Sl * fr + Sl * Sr * (ur - ul)
    ) / (Sr - Sl)


    """
    $TYPEDSIGNATURES

    Convert block-structured notation (last dim. as state variable) to matrix
    notation (first dim. as state variable). Also returns original array size.
    """
    block2mat(a::AbstractArray) = (
        (
            ndims(a) == 2 && let n = size(a, 1)
                @assert n != size(a, 2) "Too few cells for residual calculation"

                (n in (4, 5))
            end
        ) ? (a, nothing) : (
            (reshape(a, :, size(a, ndims(a))) |> permutedims), size(a)
        )
    )

    """
    $TYPEDSIGNATURES

    Convert matrix notation (first dim. as state variable) to block-structured
    notation (last dim. as state variable) given final array size
    """
    mat2block(a::AbstractMatrix, s::Union{Tuple, Nothing}) = (
        isnothing(s) ? a : (
            reshape(permutedims(a), s...)
        )
    )

    """
    $TYPEDSIGNATURES

    HLL Riemann solver flux evaluation. Receives state variable matrices
    (one row per state variable) or block-structured arrays (last dim for
    state variable) and a dimension number
    """
    function HLL(Ql::AbstractArray, Qr::AbstractArray, dim::Int64, fluid::Fluid)

        Ql, bsize = block2mat(Ql)
        Qr, _ = block2mat(Qr)

        state_l = eachrow(Ql)
        state_r = eachrow(Qr)

        prims_l = state2primitive(fluid, state_l...)
        prims_r = state2primitive(fluid, state_r...)

        pl = prims_l[1]
        Tl = prims_l[2]
        vl = prims_l[dim + 2]
        al = speed_of_sound(fluid, Tl)

        pr = prims_r[1]
        Tr = prims_r[2]
        vr = prims_r[dim + 2]
        ar = speed_of_sound(fluid, Tr)

        Fl = Ql .* vl'
        Fr = Qr .* vr'

        Fl[dim + 2, :] .+= pl
        Fr[dim + 2, :] .+= pr

        Fl[2, :] .+= pl .* vl
        Fr[2, :] .+= pr .* vr

        Sr = @. max(0.0, vl + al)
        Sl = @. min(0.0, vr - ar)

        (@. _hll(Ql, Qr, Fl, Fr, Sl', Sr')) |> x -> mat2block(x, bsize)

    end

    _Mplus(M::Real) = (
        abs(M) > 1.0 ?
        max(M, 0.0) :
        (M + 1.0) ^ 2 / 4
    )
    _Mminus(M::Real) = (
        abs(M) > 1.0 ?
        min(0.0, M) :
        - (M - 1.0) ^ 2 / 4
    )

    _pplus(M::Real, p::Real) = (
        abs(M) > 1.0 ?
        (M + abs(M)) / (2 * M) :
        (M + 1.0) / 2
    ) * p
    _pminus(M::Real, p::Real) = (
        abs(M) > 1.0 ?
        (M - abs(M)) / (2 * M) :
        (1.0 - M) / 2
    ) * p

    """
    $TYPEDSIGNATURES

    AUSM scheme flux evaluation. Receives state variable matrices
    (one row per state variable) or block-structured arrays (last dim for
    state variable) and a dimension number
    """
    function AUSM(Ql::AbstractArray, Qr::AbstractArray, dim::Int64, fluid::Fluid)

        Ql, bsize = block2mat(Ql)
        Qr, _ = block2mat(Qr)

        state_l = eachrow(Ql)
        state_r = eachrow(Qr)

        prims_l = state2primitive(fluid, state_l...)
        prims_r = state2primitive(fluid, state_r...)

        pl = prims_l[1]
        Tl = prims_l[2]
        vl = prims_l[dim + 2]
        al = speed_of_sound(fluid, Tl)

        pr = prims_r[1]
        Tr = prims_r[2]
        vr = prims_r[dim + 2]
        ar = speed_of_sound(fluid, Tr)

        ϕl = copy(Ql)
        ϕl[2, :] .+= pl

        ϕr = copy(Qr)
        ϕr[2, :] .+= pr

        Ml = vl ./ al
        Mr = vr ./ ar

        P = @. _pplus(Ml, pl) + _pminus(Mr, pr)
        Mhalf = @. _Mplus(Ml) + _Mminus(Mr)

        upwind = @. Mhalf > 0.0
        F = @. ϕl * (al * Mhalf * upwind)' + ϕr * (ar * Mhalf * (1.0 - upwind))'

        F[dim + 2, :] .+= P

        mat2block(F, bsize)

    end

    """
    $TYPEDSIGNATURES

    JST-KE scheme fluxes
    """
    function JSTKE(
        Qim1::AbstractArray{Float64},
        Qi::AbstractArray{Float64},
        Qip1::AbstractArray{Float64},
        Qip2::AbstractArray{Float64},
        dim::Int64, fluid::Fluid
    )

        Qim1, bsize = block2mat(Qim1)
        Qi, _ = block2mat(Qi)
        Qip1, _ = block2mat(Qip1)
        Qip2, _ = block2mat(Qip2)

        pim1 = state2primitive(fluid, eachrow(Qim1)...)[1]
        p, T, v = let prims = state2primitive(fluid, eachrow(Qi)...)
            (prims[1], prims[2], prims[dim + 2])
        end
        pip1, Tip1, vip1 = let prims = state2primitive(fluid, eachrow(Qip1)...)
            (prims[1], prims[2], prims[dim + 2])
        end
        pip2 = state2primitive(fluid, eachrow(Qip2)...)[1]

        a = speed_of_sound(fluid, T)
        aip1 = speed_of_sound(fluid, Tip1)

        ϵ = sqrt(eps(eltype(p)))
        ν = @. max(
                abs(pip1 + pim1 - 2 * p) / (abs(pip1 - p) + abs(pim1 - p) + ϵ),
                abs(pip2 + p - 2 * pip1) / (abs(pip1 - p) + abs(pip2 - pip1) + ϵ)
        )
        λ = @. max(
                abs(v) + a, abs(vip1) + aip1
        )

        Q = @. (Qi + Qip1) / 2
        prims = state2primitive(fluid, eachrow(Q)...)
        p = prims[1]
        v = prims[dim + 2]

        E = Q .* v'
        E[2, :] .+= (p .* v)
        E[dim + 2, :] .+= p

        (@. E + (Qi - Qip1) * (ν * λ)' / 2) |> x -> mat2block(x, bsize)

    end

    """
    $TYPEDSIGNATURES

    Obtain pressure coefficients throughout the field
    as a function of pressure throughout the field, freestream pressure
    and freestream Mach number.
    """
    function pressure_coefficient(fluid::Fluid, p, p∞::Float64, M∞::Float64)

        γ = fluid.γ

        Cp = @. 2 * (p / p∞ - 1.0) / (M∞ ^ 2 * γ)

    end

    """
    $TYPEDFIELDS

    Struct with freestream properties
    """
    struct Freestream
        fluid::Fluid
        p::Float64
        T::Float64
        v::Union{Tuple{Float64, Float64}, Tuple{Float64, Float64, Float64}}
    end

    """
    $TYPEDSIGNATURES

    Obtain `Freestream` struct from external flow conditions.
    Uses 3D flow if `β` is provided
    """
    function Freestream(
        fluid::Fluid, M∞::Float64, α::Float64, 
        β::Union{Float64, Nothing} = nothing;
        p::Float64 = 1e5, T::Float64 = 288.15
    )
        a = speed_of_sound(fluid, T)

        Freestream(
            fluid, p, T,
            (
                isnothing(β) ?
                (cosd(α), sind(α)) .* (M∞ * a) :
                (
                    cosd(α) * cosd(β), 
                    - sind(β) * cosd(α),
                    sind(α)
                ) .* (M∞ * a)
            )
        )
    end

    """
    $TYPEDSIGNATURES

    Obtain initial guess (state variables) for an N-cell mesh given
    freestream properties
    """
    initial_guess(free::Freestream, N::Int64) = primitive2state(
        free.fluid,
        fill(free.p, N), fill(free.T, N),
        [
            fill(vv, N) for vv in free.v
        ]...
    )

    _gram_schmidt(
        u::Tuple
    ) = let u = collect(u)
        nu = norm(u)

        if nu > eps(eltype(u))
            u ./= nu
        else
            u .= 0.0
            u[1] = 1.0
        end

        v = similar(u)
        v .= 0.0
        v[2] = 1.0

        v .-= (v ⋅ u) .* u

        nv = norm(v)

        if nv > eps(eltype(u))
            v ./= nv
        else
            v .= 0.0
            v[1] = 1.0
        end

        if length(u) == 2
            return [u v]
        end

        [
            u v cross(u, v)
        ]
    end

    _tocoords(M::AbstractMatrix, u, v) = (
        u .* M[1, 1] .+ v .* M[2, 1],
        u .* M[1, 2] .+ v .* M[2, 2]
    )
    _tocoords(M::AbstractMatrix, u, v, w) = (
        u .* M[1, 1] .+ v .* M[2, 1] .+ w .* M[3, 1],
        u .* M[1, 2] .+ v .* M[2, 2] .+ w .* M[3, 2],
        u .* M[1, 3] .+ v .* M[2, 3] .+ w .* M[3, 3],
    )

    _fromcoords(M::AbstractMatrix, u...) = _tocoords(M', u...)

    """
    $TYPEDSIGNATURES

    Rotate and rescale state variables to match new freestream properties
    """
    function rotate_and_rescale!(
        old::Freestream, new::Freestream, ρ::AbstractArray, E::AbstractArray, ρvs::AbstractArray...
    )
        state_old = primitive2state(
            old.fluid, old.p, old.T, old.v...
        )
        state_new = primitive2state(
            new.fluid, new.p, new.T, new.v...
        )

        Mold = _gram_schmidt(old.v)
        Mnew = _gram_schmidt(new.v)

        ρ .*= (state_new[1] / state_old[1])
        E .*= (state_new[2] / state_old[2])

        ρV_ratio = (state_new[1] / state_old[1]) * (
            norm(new.v) / (norm(old.v) + eps(Float64))
        )

        Mold = _gram_schmidt(old.v)
        Mnew = _gram_schmidt(new.v)

        for ρv in ρvs
            ρv .*= ρV_ratio
        end

        for (v, vnew) in zip(
            ρvs,
            _fromcoords(
                Mnew, _tocoords(
                    Mold, ρvs...
                )...
            )
        )
            v .= vnew
        end
    end

    """
    $TYPEDFIELDS

    Struct used for time averaging of a given property.
    Stores exponential moving average (`μ`) and its standard
    deviation (`σ`) for a moving average timescale `τ`.
    """
    mutable struct TimeAverage
        τ::Real
        μ::Any
        σ::Any
    end

    """
    $TYPEDSIGNATURES

    Constructor for a time-averaged property monitor
    """
    TimeAverage(τ::Real) = TimeAverage(τ, nothing, nothing)

    """
    $TYPEDSIGNATURES

    Add registry to a time-averaged property struct.

    Runs:

    ```
    η = dt / τ

    σ = √(σ ^ 2 * (1 - η) + (μ - Q) ^ 2 * η)
    μ = μ * (1 - η) + Q * η
    ```
    """
    function Base.push!(avg::TimeAverage, Q, dt = 1.0)

        # first registry
        if isnothing(avg.μ)
            avg.μ = copy(Q)
            avg.σ = avg.μ .* 0.0

            return avg.μ
        end

        if isa(dt, AbstractArray)
            if ndims(dt) == 1 && ndims(Q) > 1
                dt = reshape(dt, fill(1, ndims(Q) - 1)..., length(dt))
            end
        end

        η = @. dt / avg.τ

        if isa(Q, AbstractArray)
            @. avg.σ = sqrt(avg.σ ^ 2 * (1.0 - η) + (avg.μ - Q) ^ 2 * η)
            @. avg.μ = avg.μ * (1.0 - η) + Q * η
        else
            avg.σ = @. sqrt(avg.σ ^ 2 * (1.0 - η) + (avg.μ - Q) ^ 2 * η)
            avg.μ = @. avg.μ * (1.0 - η) + Q * η
        end

        avg.μ

    end

    """
    $TYPEDFIELDS

    Struct to hold a set of convergence criteria
    """
    mutable struct ConvergenceCriteria
        r0::Float64
        iterations::Int64
        rtol::Float64
        atol::Float64
        max_iterations::Int64
    end

    """
    $TYPEDSIGNATURES
    
    Constructor for convergence criteria. Convergence is reached if
    `r < r0 * rtol + atol` or `n_iterations > max_iterations`.
    """
    ConvergenceCriteria(
        ;
        max_iterations::Int64 = typemax(Int64),
        rtol::Float64 = 1e-7,
        atol::Float64 = 1e-7,
    ) = ConvergenceCriteria(
        0.0, 0, 
        rtol, atol, max_iterations
    )

    """
    $TYPEDSIGNATURES

    Register new residual array/scalar to a convergence monitor.
    The iteration count is incremented.

    Returns false if `r >= r0 * rtol + atol` and `n_iterations < max_iterations`.
    """
    function Base.push!(conv::ConvergenceCriteria, r)
        r = norm(r)

        if conv.iterations == 0
            conv.r0 = r
        end

        conv.iterations += 1
        if conv.iterations >= conv.max_iterations
            return true
        end

        if r < conv.r0 * conv.rtol + conv.atol
            return true
        end

        return false
    end

end # module CFD
