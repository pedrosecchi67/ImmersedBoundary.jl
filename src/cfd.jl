module CFD

    using DocStringExtensions

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

        u = map(ρuᵢ -> (ρuᵢ ./ ρ), ρu)

        vel = sqrt.(
            sum(
                uᵢ -> uᵢ .^ 2,
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
                uᵢ -> uᵢ .^ 2,
                u
            )
        )

        Cv = fld.R / (fld.γ - 1.0)
        et = @. ((Cv * T + vel ^ 2 / 2) * ρ)

        ρu = map(uᵢ -> ρ .* uᵢ, u)

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

    HLL Riemann solver flux evaluation. Receives state variable matrices
    (one row per state variable) and a dimension number
    """
    function HLL(Ql::AbstractMatrix, Qr::AbstractMatrix, dim::Int64, fluid::Fluid)

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

        @. _hll(Ql, Qr, Fl, Fr, Sl', Sr')

    end

end # module CFD
