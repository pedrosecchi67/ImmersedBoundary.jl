begin

@info "Flat plate test case"

wall_stl = Stereolitography(
    [0.0 1.0; 0.0 0.0], [1; 2;;]
)

h = 4e-3

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

dom = Domain(
    [0.0, 0.0], [1.0, 1.0],
    ("wall", wall_stl, h);
    hypercube_families = [
        "farfield" => [
            (1, false), (1, true), (2, true)
        ]
    ],
    verbose = true,
)

P = repeat(
    P∞'; inner = (length(dom), 1)
)

R∞ = let μ = dynamic_viscosity(fluid, P∞[2])
    ρ∞ = P∞[1] / P∞[2] / fluid.R

    μ / ρ∞ * 3.0
end
R = fill(R∞, length(dom))

Pavg = TimeAverage(1.0)
Ravg = TimeAverage(1.0)

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

    dom(P, R) do part, P, R
        let Q = primitive2state(fluid, P)
            ρ = Q[:, 1]
            T = P[:, 2]

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

            μₜ = R .* ρ

            Fv = viscous_fluxes(fluid, P, Pgrad; μₜ = μₜ)
            for dim = 1:2
                Q .+= dt .* δ(part, Fv[dim], dim)
            end

            let S = shear_rate(velocity_gradient)
                ∇S = [
                    δ(part, S, dim) for dim = 1:2
                ] |> x -> hcat(x...)
                ∇R = [
                    δ(part, R, dim) for dim = 1:2
                ] |> x -> hcat(x...)

                closure = Wray_Argawal(R, S, ∇R, ∇S)

                uvw = @view P[:, 3:end]

                ν = dynamic_viscosity(fluid, T) ./ ρ

                R .+= dt .* (
                    advection(part, uvw, R) .+ 
                    dissipation(part, ν .+ closure.νR, R) .+
                    closure.S
                )
            end

            P .= state2primitive(fluid, Q)
        end

        impose_bc!(
            part, "farfield", P, R
        ) do bdry, P, R
            Rb = similar(R)
            Rb .= R∞

            (
                freestream_bc(
                    P, bdry.normals
                ),
                Rb
            )
        end

        impose_bc!(
            part, "wall", P, R
        ) do bdry, P, R
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

            Rb = similar(R)
            @. Rb = min(
                R, res.νₜ
            )

            (
                wall_bc(
                    P, bdry.normals;
                    du!dn = res.du!dn,
                    image_distances = y
                ),
                Rb
            )
        end

        @. R = clamp(R, R∞, 100.0 * R∞)
    end

    dt̄ = dt * V∞ / 1.0

    if CTUs > 1.0
        push!(Pavg, P, dt̄)
        push!(Ravg, R, dt̄)
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

if !isnothing(Pavg.μ) # precaution for testing with fewer iterations
    P .= Pavg.μ
end
if !isnothing(Ravg.μ) # precaution for testing with fewer iterations
    R .= Ravg.μ
end

wall = dom.surfaces["wall"]
Ps = wall(P)
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
y⁺ = y .* res.uτ ./ ν

surface_data = Dict(
    "wall" => (
        Cf = Cf,
        y⁺ = y⁺,
    )
)

p, T, u, v = eachcol(P)

export_vtk(
    "flat_plate", dom;
    p = p, T = T,
    uv = [u v],
    R = R,
    surface_data = surface_data,
)

end
