stl = ibm.Stereolitography("n0012.dat")

L = 20.0

msh = ibm.Mesh(
               [-L/2,-L/2], [L,L],
               stl => 0.005;
               refinement_regions = [
                    ibm.Ball([0.0, 0.0], 0.05) => 0.001,
                    ibm.Ball([1.0, 0.0], 0.05) => 0.001,
                ],
                clipping_surface = stl,
)

mgrid = ibm.Multigrid(msh)

wall = ibm.Boundary(msh, stl)
freestream = ibm.Boundary(msh, (1, false), (1, true), (2, false), (2, true))
wall_surface = ibm.Surface(msh, stl, 0.005)

fluid = ibm.CFD.Fluid(; R = 283.0, γ = 1.4)

α = 5.0
M∞ = 0.5
T∞ = 288.15
p∞ = 1.0e5

a∞ = ibm.CFD.speed_of_sound(fluid, T∞)
u∞ = a∞ * M∞ * cosd(α)
v∞ = a∞ * M∞ * sind(α)

ρ, E, ρu, ρv = ibm.CFD.primitive2state(fluid, p∞, T∞, u∞, v∞)
ρ = fill(ρ, length(msh))
E = fill(E, length(msh))
ρu = fill(ρu, length(msh))
ρv = fill(ρv, length(msh))

Q = [ρ'; E'; ρu'; ρv']

@info "$(length(msh)) cells"

qdot = q -> begin
    E = ibm.CFD.HLL(ibm.MUSCL(q, msh, 1)..., 1, fluid)
    F = ibm.CFD.HLL(ibm.MUSCL(q, msh, 2)..., 2, fluid)

    - (ibm.∇(E, msh, 1) .+ ibm.∇(F, msh, 2))
end
timescale = (
            q;
            CFL::Float64 = 0.5,
            CFL_local::Float64 = 0.5,
)-> let (_, T, u, v) = ibm.CFD.state2primitive(fluid, eachrow(q)...)
    dx, dy = msh.spacing
    a = ibm.CFD.speed_of_sound(fluid, T)

    dt = @. min(
        dx / (a + abs(u)),
        dy / (a + abs(v)),
    ) / 2

    dtmin = minimum(dt)

    @. min(dtmin * CFL, dt * CFL_local)
end

wall_bc = ibm.BoundaryCondition() do bdry, ρ, E, ρu, ρv
    nx, ny = bdry.normals

    p, T, u, v = ibm.CFD.state2primitive(fluid, ρ, E, ρu, ρv)

    unorm = @. nx * u + ny * v

    ibm.CFD.primitive2state(
        fluid,
        p, T,
        u .- unorm .* nx,
        v .- unorm .* ny,
    )
end
freestream_bc = ibm.BoundaryCondition() do bdry, ρ, E, ρu, ρv
    nx, ny = bdry.normals

    p, T, u, v = ibm.CFD.state2primitive(fluid, ρ, E, ρu, ρv)
    a = ibm.CFD.speed_of_sound(fluid, T)

    M = @. (u * nx + v * ny) / a

    is_inlet = @. M > 0.0
    subsonic = @. M < 1.0

    p = map(
        (_p, inlet, sub) -> (
                             inlet ? (sub ? _p : p∞) : _p
        ),
        p, is_inlet, subsonic
    )
    T = map(
        (_T, inlet) -> (
                             inlet ? T∞ : _T
        ),
        T, is_inlet
    )
    u = map(
        (_u, inlet) -> (
                             inlet ? u∞ : _u
        ),
        u, is_inlet
    )
    v = map(
        (_v, inlet) -> (
                             inlet ? v∞ : _v
        ),
        v, is_inlet
    )

    ibm.CFD.primitive2state(fluid, p, T, u, v)
end
impose_bcs! = q -> begin
    ibm.impose_bc!(wall_bc, wall, eachrow(q)...)
    ibm.impose_bc!(freestream_bc, freestream, eachrow(q)...)
end
march! = (q; CFL = 100.0, CFL_local = 1.0, use_mgrid = false) -> begin
    if use_mgrid
        CFL = CFL_local
    end

    dt = timescale(q; CFL = CFL, CFL_local = CFL_local)

    # first stage of Heun's method
    dq = let _q = qdot(q) .* dt' .+ q
        impose_bcs!(_q)
        _q .- q
    end
    if use_mgrid
        dq .= mgrid(dq) .* mgrid.size_ratios'
    end
    qright = q .+ dq

    # second stage of Heun's method
    dq = let _q = qdot(qright) .* dt' .+ q
        impose_bcs!(_q)
        _q .- q
    end
    if use_mgrid
        dq .= mgrid(dq) .* mgrid.size_ratios'
    end
    qnew = (q .+ dq .+ qright) ./ 2

    dq = qnew .- q

    q .= qnew

    ibm.CFD.rms(dq)
end

for nit = 1:5000
    @time begin
        if nit < 3000
            march!(Q; use_mgrid = true)
        end
        resd = march!(Q)

        @show nit, resd
    end
end

dt = timescale(Q)
ρ, E, ρu, ρv = eachrow(Q)

p, T, u, v = ibm.CFD.state2primitive(fluid, ρ, E, ρu, ρv)
a = ibm.CFD.speed_of_sound(fluid, T)

Cp = ibm.CFD.pressure_coefficient(fluid, p, p∞, M∞)
Mach = @. sqrt(u ^ 2 + v ^ 2) / a

vtk = ibm.vtk_grid("n0012", msh;
                p = p, T = T, u = u, v = v,
                ρ = ρ, Mach = Mach, Cp = Cp,
               dt = dt)
ibm.vtk_save(vtk)

vtk = ibm.surf2vtk("surface", wall_surface;
                p = p, T = T, u = u, v = v,
                ρ = ρ, Mach = Mach, Cp = Cp)
ibm.vtk_save(vtk)

