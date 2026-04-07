begin

@info "Flat plate test case"

wall_stl = Stereolitography(
    [0.0 1.0; 0.0 0.0], [1; 2;;]
)

h = 1e-3

CFL = 0.5

fluid, P∞ = ISA_atmosphere(0.0; Mach = 0.5, û = [1.0, 0.0])
fluid = adjust_Reynolds(fluid, P∞, 1.0, 1e6)

V∞ = sum(
    P∞[3:end] .^ 2
) |> sqrt
pdyn = V∞ ^ 2 * P∞[1] / P∞[2] / fluid.R / 2

wf = WallFunction()
wall_bc = FlowBC(
    fluid, [P∞[1:2]; 0.0]; normal_flow = true
)
freestream_bc = FlowBC(
    fluid, P∞
)

msh = Mesh(
    [0.0, 0.0], [1.0, 1.0],
    ("wall", wall_stl, h);
    hypercube_families = [
        "farfield" => [
            (1, false), (1, true), (2, true)
        ]
    ],
    interior_reference = [0.5, 0.05],
    verbose = true,
)

dom = Domain(msh)

P = repeat(
    P∞'; inner = (length(dom), 1)
)

Pavg = TimeAverage(1.0)

CTUs = 0.0

march! = () -> begin
    dt = dom(P) do part, P
        _, T, u, v = eachcol(P)
        dx, dy = part.spacing |> eachcol
        a = speed_of_sound(fluid, T)

        minimum(
            (
                @. min(
                    dx / (abs(u) + a) / 2 * CFL,
                    dy / (abs(v) + a) / 2 * CFL,
                )
            )
        )
    end |> minimum

    dom(P) do part, P
        let Q = primitive2state(fluid, P)
            ρ = Q[:, 1]

            for dim = 1:2
                Pim2 = getalong(part, P, dim, -2)
                Pim1 = getalong(part, P, dim, -1)
                Pip1 = getalong(part, P, dim, 1)
                Pip2 = getalong(part, P, dim, 2)

                _, Pim12L = MUSCL(Pim2, Pim1, P)
                Pim12R, Pip12L = MUSCL(Pim1, P, Pip1)
                Pip12R, _ = MUSCL(P, Pip1, Pip2)

                dx = @view part.spacing[:, dim]

                Q .-= (
                    inviscid_fluxes(fluid, Pip12L, Pip12R, dim) .-
                    inviscid_fluxes(fluid, Pim12L, Pim12R, dim)
                ) ./ dx .* dt
            end

            Pgrad = [
                δ(part, P, dim) for dim = 1:2
            ]
            velocity_gradient = [
                Pgrad[j][:, i + 2] for i = 1:2, j = 1:2
            ]

            S = shear_rate(velocity_gradient)
            Δ = prod(part.spacing; dims = 2) |> vec |> x -> sqrt.(x)

            νSGS = Smagorinsky_νSGS(Δ, S; Cₛ = 0.21)

            Fv = viscous_fluxes(fluid, P, Pgrad; μₜ = νSGS .* ρ)
            for dim = 1:2
                Q .+= dt .* δ(part, Fv[dim], dim)
            end

            P .= state2primitive(fluid, Q)
        end

        impose_bc!(
            part, "farfield", P
        ) do bdry, P
            freestream_bc(
                P, bdry.normals
            )
        end

        impose_bc!(
            part, "wall", P
        ) do bdry, P
            p = @view P[:, 1]
            T = @view P[:, 2]
            V = let uvw = @view P[:, 3:end]
                sum(
                    uvw .^ 2; dims = 2
                ) |> vec |> x -> sqrt.(x .+ 1e-13)
            end
            μ = dynamic_viscosity(fluid, T)
            ρ = @. p / T / fluid.R

            ν = μ ./ ρ
            y = bdry.image_distances

            res = wf(y, V, ν)

            wall_bc(
                P, bdry.normals;
                du!dn = res.du!dn,
                image_distances = y
            )
        end
    end

    dt̄ = dt * V∞ / 1.0

    if CTUs > 1.0
        push!(Pavg, P, dt̄)
    end

    dt̄
end

nit = 0

println("Iteration\tCTUs")

while CTUs < 5.0 && nit < 1_000_000
    global nit, CTUs

    nit += 1
    CTUs += march!()

    println("$nit\t$CTUs")
end

P .= Pavg.μ

wall = dom.surfaces["wall"]
Ps = at_offset(wall, P)
y = wall.offsets

p = @view Ps[:, 1]
T = @view Ps[:, 2]
V = let uvw = @view Ps[:, 3:end]
    sum(
        uvw .^ 2; dims = 2
    ) |> vec |> x -> sqrt.(x .+ 1e-13)
end
μ = dynamic_viscosity(fluid, T)
ρ = @. p / T / fluid.R

ν = μ ./ ρ

res = wf(y, V, ν)

Cf = res.uτ .^ 2 .* ρ ./ pdyn

surface_data = Dict(
    "wall" => (
        Cf = Cf,
    )
)

p, T, u, v = eachcol(P)

export_vtk(
    "flat_plate", dom;
    p = p, T = T,
    uv = [u v],
    surface_data = surface_data,
)

end
