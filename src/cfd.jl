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
        k::AbstractVector
        μref::Real
        Tref::Real
        S::Real
    end

    """
    $TYPEDSIGNATURES

    Constructor for a fluid (defaults to air).

    If the thermal conductivity is `k`, the thermal conductivity is considered tempterature
    dependent as per the coefficients of a polynomial:

    ```
    k = 0.0
    for (i, ki) in enumerate(fluid.k)
        k += ki * T ^ (i - 1)
    end
    ```

    The other arguments are used for Sutherland's law.
    """
    Fluid(
        ;
        R::Real = 283.0,
        γ::Real = 1.4,
        k::Union{Real, AbstractVector} = [0.00646, 6.468e-5],
        μref::Real = 1.716e-5,
        Tref::Real = 273.15,
        S::Real = 110.4,
    ) = Fluid(
        R, γ, (
            k isa Real ? 
            [k] : copy(k)
        ), μref, Tref, S
    )

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

    Obtain viscosity from Sutherland's law
    """
    dynamic_viscosity(
        fld::Fluid, T
    ) = (
        @. fld.μref * ((T / fld.Tref) ^ (2.0 / 3)) * (fld.Tref + fld.S) / (T + fld.S)
    )

    """
    $TYPEDSIGNATURES

    Obtain heat conductivity given temperature
    """
    function heat_conductivity(fld::Fluid, T)
        k = @. 0.0 * T
        for (i, ki) in enumerate(fld.k)
            k += @. ki * T ^ (i - 1)
        end
        k
    end

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

    Turn array of state variables to array of primitive variables
    """
    state2primitive(fld::Fluid, Q::AbstractArray) = let (q, s) = block2mat(Q)
        p = similar(q)
        for (prim, row) in zip(
            state2primitive(fld, eachrow(q)...), eachrow(p)
        )
            row .= prim
        end
        mat2block(p, s)
    end

    """
    $TYPEDSIGNATURES

    Turn array of primitive variables to array of state variables
    """
    primitive2state(fld::Fluid, P::AbstractArray) = let (p, s) = block2mat(P)
        q = similar(p)
        for (stat, row) in zip(
            primitive2state(fld, eachrow(p)...), eachrow(q)
        )
            row .= stat
        end
        mat2block(q, s)
    end

    """
    $TYPEDSIGNATURES

    HLL Riemann solver flux evaluation. Receives primitive variable matrices
    (one row per prim. variable) or block-structured arrays (last dim for
    prim. variable) and a dimension number
    """
    function HLL(Pl::AbstractArray, Pr::AbstractArray, dim::Int64, fluid::Fluid)

        Pl, bsize = block2mat(Pl)
        Pr, _ = block2mat(Pr)

        prims_l = eachrow(Pl)
        prims_r = eachrow(Pr)

        Ql = primitive2state(fluid, Pl)
        Qr = primitive2state(fluid, Pr)

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

    AUSM scheme flux evaluation. Receives primitive variable matrices
    (one row per prim. variable) or block-structured arrays (last dim for
    prim. variable) and a dimension number
    """
    function AUSM(Pl::AbstractArray, Pr::AbstractArray, dim::Int64, fluid::Fluid)

        Pl, bsize = block2mat(Pl)
        Pr, _ = block2mat(Pr)

        prims_l = eachrow(Pl)
        prims_r = eachrow(Pr)

        Ql = primitive2state(fluid, Pl)
        Qr = primitive2state(fluid, Pr)

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

    JST-KE scheme fluxes from primitive variable arrays
    """
    function JSTKE(
        Pim1::AbstractArray{Float64},
        Pi::AbstractArray{Float64},
        Pip1::AbstractArray{Float64},
        Pip2::AbstractArray{Float64},
        dim::Int64, fluid::Fluid;
        νmin::Union{AbstractArray{Float64}, Nothing} = nothing,
        νmin_ip1::Union{AbstractArray{Float64}, Nothing} = nothing,
    )

        Pim1, bsize = block2mat(Pim1)
        Pi, _ = block2mat(Pi)
        Pip1, _ = block2mat(Pip1)
        Pip2, _ = block2mat(Pip2)
        if isnothing(νmin)
            νmin = 0.0
        else
            νmin, _ = block2mat(νmin)
        end
        if isnothing(νmin_ip1)
            νmin_ip1 = 0.0
        else
            νmin_ip1, _ = block2mat(νmin_ip1)
        end

        pim1 = @view Pim1[1, :]
        p = @view Pi[1, :]
        T = @view Pi[2, :]
        v = @view Pi[2 + dim, :]
        pip1 = @view Pip1[1, :]
        Tip1 = @view Pip1[2, :]
        vip1 = @view Pip1[2 + dim, :]
        pip2 = @view Pip2[1, :]

        Qi = primitive2state(fluid, Pi)
        Qip1 = primitive2state(fluid, Pip1)

        a = speed_of_sound(fluid, T)
        aip1 = speed_of_sound(fluid, Tip1)

        ϵ = sqrt(eps(eltype(p)))
        ν = @. max(
                abs(pip1 + pim1 - 2 * p) / (abs(pip1 - p) + abs(pim1 - p) + ϵ),
                abs(pip2 + p - 2 * pip1) / (abs(pip1 - p) + abs(pip2 - pip1) + ϵ),
                νmin, νmin_ip1
        )
        λ = @. max(
                abs(v) + a, abs(vip1) + aip1
        )

        P = @. (Pi + Pip1) / 2
        Q = primitive2state(fluid, P)

        p = @view P[1, :]
        v = @view P[dim + 2, :]

        E = Q .* v'
        E[2, :] .+= (p .* v)
        E[dim + 2, :] .+= p

        (@. E + (Qi - Qip1) * (ν * λ)' / 2) |> x -> mat2block(x, bsize)

    end

    """
    $TYPEDSIGNATURES

    Obtain viscous and conductive fluxes given array of primitive variables,
    fluid struct and components of the primitive variable array gradient along each axis.
    """
    function viscous_fluxes(P::AbstractArray, fluid::Fluid, Pgrad::AbstractArray...;
        μt::Union{Real, AbstractArray} = 0.0)
        
        P, bsize = block2mat(P)
        if μt isa AbstractArray
            μt, _ = block2mat(μt)
            μt = vec(μt)
        end
        Pgrad = [
            block2mat(pgrad)[1] for pgrad in Pgrad
        ]

        T = @view P[2, :]
        vels = eachrow(P)[3:end]
        μ = dynamic_viscosity(fluid, T) .+ μt
        k = heat_conductivity(fluid, T)

        nd = length(vels)

        # calculate stresses
        velgrad = [view(Pgrad[j], i + 2, :) for i = 1:nd, j = 1:nd]
        divu = sum(
            i -> velgrad[i, i], 1:nd
        )

        τ = [
            (
                velgrad[i, j] .+ velgrad[j, i] .- (
                    i == j ? divu .* (2.0 / 3) : 0.0
                )
            ) .* μ for i = 1:nd, j = 1:nd
        ]

        # calculate heat flux
        f = [
            pgrad[2, :] .* k for pgrad in Pgrad
        ]

        # compile along each axis
        fluxes = AbstractArray[]
        for i = 1:nd
            F = similar(P)
            F .= 0.0

            F[2, :] .+= f[i] # add heat flux

            # add shear contribution to energy
            for j = 1:nd
                F[2, :] .+= vels[j] .* τ[i, j]
            end

            # add viscous fluxes to momenta
            for j = 1:nd
                F[2 + j, :] .+= τ[i, j]
            end

            push!(
                fluxes, mat2block(F, bsize)
            )
        end

        fluxes

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

    """
    $TYPEDFIELDS

    Struct to hold a CTU counter
    """
    mutable struct CTUCounter
        adimensional_time::Float64
        L::Float64
        λ::Float64
    end

    """
    $TYPEDSIGNATURES

    Constructor for a CPU counter.
    Counts `V × t / L` if `count_speed_of_sound = false`
    or `(V + a) × t / L` otherwise.

    `freestream` may be a scalar or a `Freestream` struct.
    If `freestream` is a scalar, it is considered to be the
    characteristic velocity of the flow.
    """
    function CTUCounter(
        L::Float64, freestream;
        count_speed_of_sound::Bool = false
    )
        λ = freestream
        if freestream isa Freestream
            λ = norm(freestream.v)

            if count_speed_of_sound
                λ += speed_of_sound(
                    freestream.fluid, freestream.T
                )
            end
        end

        CTUCounter(0.0, L, λ)
    end

    """
    $TYPEDSIGNATURES

    Add time step to CTU counter and return the resulting
    CTU count
    """
    Base.push!(cnt::CTUCounter, dt) = let dtmin = minimum(dt)
        cnt.adimensional_time += dtmin * cnt.λ / cnt.L

        cnt.adimensional_time
    end

    """
    $TYPEDSIGNATURES

    Reduce value of `dt` for each cell until no `NaN` or `Inf` is found.

    Done in-place. Returns final residual, the number of cells with timestep reductions and
    the maximum time-step reduction factor.

    Function `f(dt, Q, args...; kwargs...)` should return `dQ!dt`, 
    where `Q` is a state variable matrix with shape `(nvars, ncells)` or
    `(ncells, nvars)`.

    If `check_residuals = true`, both `Qnew` and `f(dt, Qnew)`, for 
    `Qnew = Q .+ f(dt, Q) .* dt`, are checked for violations.
    """
    function clip_CFL!(
        f,
        dt::AbstractVector{Float64}, Q::AbstractMatrix{Float64}, 
        args::AbstractMatrix{Float64}...;
        reduction_ratio::Real = 0.5,
        check_residuals::Bool = false,
        max_iterations::Int = 10,
        lower_boundary::Union{AbstractVector{Float64}, Float64} = -Inf64,
        upper_boundary::Union{AbstractVector{Float64}, Float64} = Inf64,
        kwargs...
    )
        min_ratio = 1.0

        r = q -> f(dt, q, args...; kwargs...)

        # check if input is in column-major order
        col_major = false
        if size(Q, 1) < length(dt)
            dt = dt'
            col_major = true
        else
            lower_boundary = lower_boundary'
            upper_boundary = upper_boundary'
        end

        is_valid = u -> let iv = any(
            uu -> !(isnan(uu) || isinf(uu)), u;
            dims = (col_major ? 1 : 2)
        )
            if !(isinf(lower_boundary) && isinf(upper_boundary))
                iv .= iv .&& any(
                    (@. u >= lower_boundary && u <= upper_boundary);
                    dims = (col_major ? 1 : 2)
                )
            end

            iv
        end

        reduced = falses(length(dt))

        Qnew = copy(Q)
        for _ = 1:max_iterations
            Qnew .= Q .+ r(Q) .* dt
            iv = is_valid(Qnew)

            if all(iv)
                break
            end

            if check_residuals
                iv .= iv .&& is_valid(r(Qnew))
            end

            # reduce timestep for invalid cells
            @. dt = dt * max(iv, reduction_ratio)
            @. reduced = reduced || (!iv)
            min_ratio *= reduction_ratio
        end

        # to save memory:
        residual = Qnew
        residual .= (Qnew .- Q) ./ dt

        (residual, sum(reduced), min_ratio)
    end

end # module CFD
