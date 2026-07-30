begin
    @info "Running advection test case"

    lower = Stereolitography(
        [0.0 1.0; 0.0 0.0]
    )
    upper = Stereolitography(
        [0.0 0.0; 0.0 1.0]
    )

    msh = Mesh(
        [0.0, 0.0], [1.0, 1.0],
        ("lower", lower, 1f-2),
        ("upper", upper, 1f-2);
        refinement_regions = [
            Line([0.0, 0.0], [1.0, 1.0]) => 2f-2,
            Line([0.0, 0.0], [0.5, 0.5]) => 1f-2,
        ],
        verbose = true,
    )

    dom = Domain(msh; verbose = true,
        hypercube_families = [
            "outlet" => [(1, true), (2, true)],
        ],
    )

    u = zeros(Float32, length(dom))

    apply_bcs! = u -> begin
        impose_bc!(
            dom, "upper", u
        ) do bdry, u
            1.0f0
        end
        impose_bc!(
            dom, "lower", u
        ) do bdry, u
            0.0f0
        end
        impose_bc!(
            dom, "outlet", u
        ) do bdry, u
            copy(u)
        end
    end

    Cx = ones(Float32, length(dom))
    Cy = ones(Float32, length(dom))
    C = [Cx Cy]

    timestep_length = () -> dom() do part
        0.5f0 / maximum(
            max.(
                unsigned_green_gauss(part, at_faces(part, Cx, 1), 1),
                unsigned_green_gauss(part, at_faces(part, Cy, 2), 2),
            )
        )
    end |> minimum

    march! = u -> begin
        ud = similar(u)
        ud .= 0

        dt = timestep_length() * 0.75f0

        dom(u, ud) do part, u, ud
            for dim = 1:ndims(part)
                Cd = @view C[:, dim]
                Cf = at_faces(part, Cd, dim)

                ∇u = cell_gradient(part, u, dim)
                uL, uR = MUSCL(part, u, ∇u, dim)

                ud .-= green_gauss(
                    part,
                    (@. (uL + uR) * Cf / 2 + abs(Cf) * (uL - uR) / 2),
                    dim
                )
            end
        end

        u .+= ud .* dt
        apply_bcs!(u)

        u
    end

    for _ = 1:1000
        march!(u)
    end

    export_vtk("advection", dom; u = u)
end
