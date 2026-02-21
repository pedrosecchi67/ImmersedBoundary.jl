begin

@info "Running RAE-2822 test case"

stl = Stereolitography("rae2822.dat")
features = feature_regions(stl; radius = 0.05)
feature_dfield = DistanceField(features)

msh = Mesh(
    [-40.0, -40.0],
    [80.0, 80.0],
    ("wall", stl, 4e-3);
    refinement_regions = [
        (feature_dfield, 0.5e-3),
    ],
    interior_reference = [2.0, 2.0],
    hypercube_families = [
        "freestream" => [(1, false), (1, true), (2, false), (2, true)],
    ],
    verbose = true
)

dom = Domain(msh;
    max_partition_size = length(msh) ÷ 2)

M∞ = 0.72
fluid, P∞ = ISA_atmosphere(
    0.0;
    û = streamwise_direction(5.0),
    Mach = M∞
)
P = repeat(P∞'; inner = (length(dom), 1))

fluid = adjust_Reynolds(fluid, P∞, 1.0, 10_000.0)
@show fluid.μref

freestream = FlowBC(
    fluid, P∞
)
wall = FlowBC(
    fluid, [0.0, 0.0, 0.0, 0.0];
    normal_flow = false
)

k = zeros(length(dom))

for _ = 1:2000
    dt = dom(P) do part, P
        dt = Inf64

        a = speed_of_sound(fluid, P[:, 2])
        for dim = 1:2
            u = @view P[:, 2 + dim]
            dx = @view part.spacing[:, dim]

            dt = min(
                dt,
                minimum(dx ./ (abs.(u) .+ a)) * 0.25
            )
        end

        dt
    end |> minimum

    dom(P, k) do part, P, k
        let Qnew = primitive2state(fluid, P)        
            Pgrad = [
                δ(part, P, dim) for dim = 1:2
            ]

            Fv = viscous_fluxes(fluid, P, Pgrad)

            for dim = 1:2
                Pim2 = getalong(part, P, dim, -2)
                Pim1 = getalong(part, P, dim, -1)
                Pip1 = getalong(part, P, dim, 1)
                Pip2 = getalong(part, P, dim, 2)

                _, Pim12L = MUSCL(Pim2, Pim1, P)
                Pim12R, Pip12L = MUSCL(Pim1, P, Pip1)
                Pip12R, _ = MUSCL(P, Pip1, Pip2)

                Qnew .-= dt .* (
                    inviscid_fluxes(fluid, Pip12L, Pip12R, dim) .-
                    inviscid_fluxes(fluid, Pim12L, Pim12R, dim)
                ) ./ part.spacing[:, dim]

                Qnew .+= dt .* δ(
                    part, Fv[dim], dim
                )
            end

            P .= state2primitive(fluid, Qnew)
        end

        let vels = @view P[:, 3:end]
            k .+= dt .* (
                advection(part, vels, k) .+
                dissipation(part, 1e-2, k)
            )
        end

        impose_bc!(part, "freestream", P) do bdry, P
            freestream(P, bdry.normals)
        end

        impose_bc!(part, "wall", P) do bdry, P
            wall(P, bdry.normals)
        end

        impose_bc!(part, "wall", k) do bdry, k
            kb = similar(k)
            kb .= 1.0

            kb
        end
    end
end

p, T, u, v = eachcol(P)
Cp = pressure_coefficient(fluid, p, P∞[1], M∞)

dom = Domain(msh;
    max_partition_size = length(msh) ÷ 2)

export_vtk(
    "rae2822", dom;
    p = p, T = T, uv = [u v], Cp = Cp, k = k,
)

end
