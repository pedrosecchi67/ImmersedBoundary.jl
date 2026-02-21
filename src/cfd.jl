module CFD

    using LinearAlgebra

    using DocStringExtensions

    export Fluid

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

    export speed_of_sound, dynamic_viscosity, heat_conductivity

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
    ) = let T = @. clamp(T, 10.0, Inf64)
        (
            @. fld.μref * ((T / fld.Tref) ^ (2.0 / 3)) * (fld.Tref + fld.S) / (T + fld.S)
        )
    end

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

    export primitive2state, state2primitive

    """
    $TYPEDSIGNATURES

    Obtain state variables from primitive variables.
    
    Both are given as matrices in format:
    
    ```
    p, T, u, v[, w] = eachcol(P)
    ρ, E, ρu, ρv[, ρw] = eachcol(Q)
    ```
    """
    function primitive2state(
        fluid::Fluid, P::AbstractMatrix
    )
        p = @view P[:, 1]
        T = clamp.(P[:, 2], 10.0, Inf64)
        u = @view P[:, 3:end]

        k = (sum(u .^ 2; dims = 2) ./ 2) |> vec

        ρ = @. p / (fluid.R * T)
        E = @. ρ * (
            fluid.R / (fluid.γ - 1.0) * T + k
        )

        [
            ρ E (ρ .* u)
        ]
    end

    """
    $TYPEDSIGNATURES

    Obtain primitive variables from state variables.
    
    Both are given as matrices in format:
    
    ```
    p, T, u, v[, w] = eachcol(P)
    ρ, E, ρu, ρv[, ρw] = eachcol(Q)
    ```
    """
    function state2primitive(
        fluid::Fluid, Q::AbstractMatrix
    )
        ρ = @view Q[:, 1]
        E = @view Q[:, 2]
        ρu = @view Q[:, 3:end]

        u = ρu ./ ρ
        k = (sum(u .^ 2; dims = 2) ./ 2) |> vec

        p = @. (fluid.γ - 1.0) * (E - ρ * k)
        T = @. clamp(p / (ρ * fluid.R), 10.0, Inf64)

        [p T u]
    end

    export FlowBC

    """
    $TYPEDFIELDS

    Struct to define a generic boundary condition formulation
    """
    struct FlowBC
        fluid::Fluid
        P::AbstractVector
        normal_flow::Bool
    end

    """
    $TYPEDSIGNATURES

    Constructor for a flow boundary condition.
    If `normal_flow` is defined, the BC is not considered to be a Dirichlet condition,
    but instead a Neumann/Robin condition for the parallel velocity component,
    and the last velocity in `P` is imposed in the direction normal to the boundary.

    Example:

    ```
    # P in [p T u v [w]] format
    p∞, T∞, u∞, v∞ = 1e5, 288.15, 100.0, 10.0

    inlet_outlet = FlowBC(fluid, [p∞, T∞, u∞, v∞])

    slip_wall = FlowBC(fluid, [p∞, T∞, 0.0]; # single vel. component: normal
        normal_flow = true)

    no_slip_wall = FlowBC(fluid, [p∞, T∞, 0.0, 0.0])

    domain(P) do part, P
        # example for freestream:

        impose_bc!(part, "freestream", P) do bdry, P
            inlet_outlet(
                P, bdry.normals
            )
        end

        # example for Euler wall:

        impose_bc!(part, "euler_wall", P) do bdry, P
            slip_wall(
                P, bdry.normals
            )
        end

        # example for viscous wall:

        # calculate du!dn at wall somehow
        impose_bc!(part, "neumann_wall", P) do bdry, P
            slip_wall(
                P, bdry.normals;
                du!dn = du!dn,
                image_distances = bdry.image_distances
            )
        end

        # example for viscous wall (laminar):

        impose_bc!(part, "laminar_wall", P) do bdry, P
            no_slip_wall(
                P, bdry.normals
            )
        end
    end
    ```
    """
    FlowBC(
        fluid::Fluid,
        P::AbstractVector;
        normal_flow::Bool = false
    ) = FlowBC(
        fluid, P, normal_flow
    )

    """
    $TYPEDSIGNATURES

    Impose BCs to primitive variables.
    See docstring for `FlowBC` constructor for examples.
    """
    function (bc::FlowBC)(
        P::AbstractMatrix, normals::AbstractMatrix;
        image_distances::Union{Nothing, AbstractVector} = nothing,
        du!dn::Union{Nothing, AbstractVector} = nothing,
    )
        p∞ = bc.P[1]
        T∞ = bc.P[2]
        u∞ = bc.P[3:end]

        if bc.normal_flow
            @assert length(bc.P) == 3 "Only 3 parcels (p, T and normal flow) allowed for normal_flow = true BC"
            un = similar(P, (size(P, 1),))
            un .= u∞[1]
        else
            un = normals * u∞
        end

        p = @view P[:, 1]
        T = @view P[:, 2]
        u = @view P[:, 3:end]
        current_un = sum(u .* normals; dims = 2) |> vec

        a = speed_of_sound(bc.fluid, T)

        M = @. abs(un) / a

        # note that the >[=]s here indicate that walls (un = 0.0)
        # will have appropriate BCs
        pb = @. (un >= 0.0) * (
            (M > 1.0) * p∞ + (M <= 1.0) * p
        ) + (un < 0.0) * (
            (M > 1.0) * p + (M <= 1.0) * p∞
        )
        Tb = @. (un > 0.0) * T∞ + (un <= 0.0) * T

        ub = nothing
        if bc.normal_flow
            ub = u .+ normals .* (
                un .- current_un
            )
        else
            ub = (@. un < 0.0) .* u .+ (@. un >= 0.0) .* u∞'
        end

        if isnothing(du!dn) != isnothing(image_distances)
            throw(error("du!dn and image_distances must be passed together for BC imposition"))
        end

        if !isnothing(du!dn)
            ϵ = eps(Float64) |> sqrt
            V = sum(ub .^ 2; dims = 2) |> vec |> x -> sqrt.(x) .+ ϵ

            @. ub *= (V - du!dn * image_distances) / V
        end

        [pb Tb ub]
    end

    export ISA_atmosphere

    function _ISA_atmosphere(altitude_m::Real, ΔT::Real = 0.0)
        # Constants
        R = 287.05287  # Specific gas constant for dry air [J/(kg·K)]
        g0 = 9.80665   # Gravitational acceleration [m/s²]
        
        # Sea level conditions
        P0 = 101325.0   # Pressure at sea level [Pa]
        T0 = 288.15     # Temperature at sea level [K]
        
        # Layer definitions [base_altitude, base_temp, lapse_rate, base_pressure]
        # Lapse rate in K/km, converted to K/m internally
        layers = [
            (0.0,      288.15, -6.5,   101325.0),    # Troposphere
            (11000.0,  216.65,  0.0,   22632.0),     # Tropopause
            (20000.0,  216.65,  1.0,   5474.9),      # Stratosphere 1
            (32000.0,  228.65,  2.8,   868.02),      # Stratosphere 2
            (47000.0,  270.65,  0.0,   110.91),      # Stratopause
            (51000.0,  270.65, -2.8,   66.939),      # Mesosphere 1
            (71000.0,  214.65, -2.0,   3.9564)       # Mesosphere 2
        ]
        # Note: Above 86km, the model becomes more complex and not included here
        
        # Check if altitude is within valid range
        if altitude_m < 0
            error("Altitude cannot be negative")
        elseif altitude_m > 86000
            @warn "Altitude above 86 km - model accuracy decreases"
        end
        
        # Find the appropriate layer
        layer_idx = 1
        for i in 1:length(layers)-1
            if altitude_m >= layers[i][1]
                layer_idx = i
            end
        end
        
        # Get layer parameters
        h_base, T_base, lapse_rate, P_base = layers[layer_idx]
        lapse_rate_per_m = lapse_rate / 1000.0  # Convert to K/m
        
        # Calculate geometric height difference
        delta_h = altitude_m - h_base
        
        # Calculate temperature with offset
        # Note: ΔT is applied uniformly at all altitudes
        T = T_base + lapse_rate_per_m * delta_h + ΔT
        
        # Calculate pressure based on lapse rate
        if abs(lapse_rate_per_m) < 1e-10
            # Isothermal layer - using T_base (without offset) for pressure calculation
            # This maintains hydrostatic consistency
            P = P_base * exp(-g0 * delta_h / (R * (T_base + ΔT)))
        else
            # Gradient layer
            exponent = -g0 / (R * lapse_rate_per_m)
            # Use temperatures with offset for pressure calculation
            T_base_offset = T_base + ΔT
            T_offset = T_base_offset + lapse_rate_per_m * delta_h
            P = P_base * (T_offset / T_base_offset)^exponent
        end
        
        return (P, T)
    end

    """
    $TYPEDSIGNATURES

    Return fluid and primitive variables using ISA atmosphere.
    Allows for Mach number definition and takes a freestream direction
    vector `û` to account for α and β.

    If `V` is provided, the Mach number is disregarded and a given velocity
    is imposed.
    """
    ISA_atmosphere(
        altitude_m::Float64; 
        ΔT::Float64 = 0.0,
        Mach::Float64 = 0.0,
        V::Union{Float64, Nothing} = nothing,
        û::AbstractVector = [1.0],
    ) = let (p, T) = _ISA_atmosphere(altitude_m, ΔT)
        fluid = Fluid()

        u = V
        if isnothing(u)
            a = speed_of_sound(fluid, T)
            u = Mach * a
        end

        û = û ./ (eps(Float64) + norm(û))

        (fluid, [p; T; (u .* û)])
    end

    export streamwise_direction

    """
    $TYPEDSIGNATURES

    Obtain direction of the flow (normal vector) as a function
    of `α`, in 2D.
    """
    streamwise_direction(α::Real) = [
        cosd(α), sind(α)
    ]

    export pressure_coefficient

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
    $TYPEDSIGNATURES

    Obtain direction of the flow (normal vector) as a function
    of `α, β`, in 3D.
    """
    streamwise_direction(α::Real, β::Real) = [
        cosd(α) * cosd(β), - cosd(α) * sind(β), sind(α)
    ]

    _cusp_f1(m::Real, a0::Real) = (
        (abs(m) < 1.0) * (a0 + (1.5 - 2 * a0) * m ^ 2 + (a0 - 0.5) * m ^ 4) +
        (abs(m) >= 1.0) * abs(m)
    )

    _cusp_f2(m::Real) = (
        (abs(m) < 1.0) * (m * (3.0 - m ^ 2) / 2) + 
        (abs(m) >= 1.0) * sign(m)
    )

    export inviscid_fluxes

    """
    $TYPEDSIGNATURES

    Obtain CUSP scheme inviscid fluxes given primitive variables on the left and right sides
    of a face
    """
    function inviscid_fluxes(
        fluid::Fluid, PL::AbstractMatrix, PR::AbstractMatrix, dim::Int;
        a0::Real = 0.25,
    )
        UcL = primitive2state(fluid, PL)
        pL = PL[:, 1]
        UcL[:, 2] .+= pL # pressure parcel

        UcR = primitive2state(fluid, PR)
        pR = PR[:, 1]
        UcR[:, 2] .+= pR # pressure parcel

        P = @. (PL + PR) / 2
        T = @view P[:, 2]
        u = @view P[:, 2 + dim]

        a = speed_of_sound(fluid, T)
        M = @. u / a

        F = @. u * (UcL + UcR) / 2 - _cusp_f1(M, a0) * a * (UcR - UcL) / 2
        Fmom = @view F[:, 2 + dim]

        @. Fmom += ((pR + pL) / 2 - _cusp_f2(M) * (pR - pL) / 2)

        F
    end

    export Reynolds_number, adjust_Reynolds

    """
    $TYPEDSIGNATURES

    Get Reynolds number given a fluid, primitive variables and a reference length
    """
    function Reynolds_number(
        fluid::Fluid, P∞::AbstractVector, Lref::Real
    )
        V = P∞[3:end] |> norm
        T = P∞[2]
        p = P∞[1]

        ρ = @. p / (fluid.R * T)
        μ = dynamic_viscosity(fluid, T)

        V * Lref * ρ / μ
    end

    """
    $TYPEDSIGNATURES

    Adjust Reynolds number by editing fluid reference viscosity and returning a new 
    fluid struct.
    """
    function adjust_Reynolds(
        fluid::Fluid, P∞::AbstractVector, Lref::Real, Re::Real
    )
        Re_old = Reynolds_number(fluid, P∞, Lref)
        μref = fluid.μref * Re_old / Re

        Fluid(
            fluid.R, fluid.γ, fluid.k, μref, fluid.Tref, fluid.S
        )
    end

    export viscous_fluxes

    """
    $TYPEDSIGNATURES

    Obtain inviscid fluxes given a fluid, primitive variables and primitive variable gradients
    (a vector of matrices, with each matrix corresponding to the gradient along one axis).
    Returns fluxes along all Cartesian dimensions.

    Example:

    ```
    Pgrad = [
        δ(part, P, dim) for dim = 1:3
    ]
    
    Fvx, Fvy, Fvz = viscous_fluxes(fluid, P, Pgrad)
    ```
    """
    function viscous_fluxes(
        fluid::Fluid, P::AbstractMatrix, Pgrad::AbstractVector;
        μₜ::Union{AbstractVector, Real} = 0.0
    )
        T = @view P[:, 2]

        μ = dynamic_viscosity(fluid, T) .+ μₜ
        k = heat_conductivity(fluid, T)

        nd = size(P, 2) - 2

        vel_grad = [
            Pgrad[j][:, 2 + i] for i = 1:nd, j = 1:nd
        ]
        vels = [
            P[:, 2 + dim] for dim = 1:nd
        ]

        # shear tensor
        divu = similar(T)
        divu .= 0.0
        for i = 1:nd
            divu .+= vel_grad[i, i]
        end

        τ = [
            (
                (vel_grad[i, j] .+ vel_grad[j, i]) .- (i == j ? 2.0 / 3.0 : 0.0) .* divu
            ) .* μ for i = 1:nd, j = 1:nd
        ]

        # heat fluxes
        f = [
            pgrad[:, 2] .* k for pgrad in Pgrad
        ]

        [
            let F = similar(P)
                F .= 0.0

                # energy
                F[:, 2] .+= f[dim]
                for j = 1:nd
                    F[:, 2] .+= τ[dim, j] .* vels[j]
                end

                # momentum
                for j = 1:nd
                    F[:, 2 + j] .+= τ[dim, j]
                end

                F
            end for dim = 1:nd
        ]
    end

    export TimeAverage

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

end
