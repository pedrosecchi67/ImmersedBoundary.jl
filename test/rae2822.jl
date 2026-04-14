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

M∞ = 0.72f0
fluid, P∞ = ISA_atmosphere(
    0.0f0;
    û = streamwise_direction(5.0f0),
    Mach = M∞
)
P = repeat(P∞'; inner = (length(dom), 1))

fluid = adjust_Reynolds(fluid, P∞, 1.0f0, 10_000.0f0)
@show fluid.μref

freestream = FlowBC(
    fluid, P∞
)
wall = FlowBC(
    fluid, [0.0f0, 0.0f0, 0.0f0, 0.0f0];
    normal_flow = false
)

k = zeros(Float32, length(dom))

νSGS = similar(k)
νSGS .= 0

@assert eltype(P) === Float32

for _ = 1:100
    dt = dom(P) do part, P
        @assert eltype(part.spacing) === Float32
        @assert eltype(part.centers) === Float32

        dt = Inf32

        a = speed_of_sound(fluid, P[:, 2])
        for dim = 1:2
            u = @view P[:, 2 + dim]
            dx = @view part.spacing[:, dim]

            dt = min(
                dt,
                minimum(dx ./ (abs.(u) .+ a)) * 0.25f0
            )
        end

        dt
    end |> minimum

    @assert dt isa Float32

    dom(P, k, νSGS) do part, P, k, νSGS
        let Qnew = primitive2state(fluid, P)        
            Pgrad = [
                δ(part, P, dim) for dim = 1:2
            ]

            nd = length(Pgrad)
            let S = shear_rate(
                [
                    Pgrad[j][:, i + 2] for i = 1:nd, j = 1:nd
                ]
            )
                Δ = prod(
                    part.spacing; dims = 2
                ) |> vec |> x -> x .^ (1.0f0 / nd)

                νSGS .= Smagorinsky_νSGS(Δ, S)
            end

            Fv = viscous_fluxes(fluid, P, Pgrad)

            for dim = 1:2
                @assert eltype(Fv[dim]) === Float32

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

        impose_bc!(part, "wall", k) do bdry, ki
            kb = similar(ki)
            kb .= 1

            k_at_bdry = bdry(k)

            kb
        end
    end
end

p, T, u, v = eachcol(P)
Cp = pressure_coefficient(fluid, p, P∞[1], M∞)

uvdiv = similar(p)
uvdiv .= 0
dom(uvdiv, P[:, 3:end]) do part, uvdiv, uv
    uvdiv .= divergent(part, uv)
end

export_vtk(
    "rae2822", dom;
    p = p, T = T, uv = [u v], Cp = Cp, k = k,
    uvdiv = uvdiv, νSGS = νSGS
)

end
