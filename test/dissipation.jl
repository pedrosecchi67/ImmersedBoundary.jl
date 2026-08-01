begin
    @info "Running dissipation test case"

    lower = Stereolitography(
        [0.0 1.0; 0.0 0.0]
    )
    upper = Stereolitography(
        [0.0 0.0; 0.0 1.0]
    )

    msh = Mesh(
        [0.0, 0.0], [1.0, 1.0],
        ("lower", lower, 2f-2),
        ("upper", upper, 2f-2);
        refinement_regions = [
            Line([0.0, 0.0], [1.0, 1.0]) => 4f-2,
            Line([0.0, 0.0], [0.5, 0.5]) => 2f-2,
        ],
        verbose = true,
    )

    dom = Domain(msh; verbose = true,
        hypercube_families = [
            "neumann" => [(1, true), (2, true)],
        ],
    )

    uv = zeros(Float32, (length(dom), 2))

    apply_bcs! = uv -> begin
        impose_bc!(
            dom, "upper", uv
        ) do bdry, uv
            uvb = similar(uv)
            uvb .= [1.0f0, 0.0f0]'

            uvb
        end
        impose_bc!(
            dom, "lower", uv
        ) do bdry, uv
            uvb = similar(uv)
            uvb .= [0.0f0, 1.0f0]'

            uvb
        end
        impose_bc!(
            dom, "neumann", uv
        ) do bdry, uv
            copy(uv)
        end
    end

    timestep_length = () -> dom() do part
        1.0f0 / maximum(
            (
                unsigned_green_gauss(part, 1.0f0 ./ face_distance(part, 1), 1) .+
                unsigned_green_gauss(part, 1.0f0 ./ face_distance(part, 2), 2)
            )
        )
    end |> minimum

    march! = uv -> begin
        uvd = similar(uv)
        uvd .= 0

        dt = timestep_length() * 0.5f0

        dom(uv, uvd) do part, uv, uvd
            for dim = 1:ndims(part)
                uvd .+= green_gauss(
                    part,
                    face_gradient(part, uv, dim),
                    dim
                )
            end
        end

        uv .+= uvd .* dt
        apply_bcs!(uv)

        uv
    end

    for _ = 1:1000
        march!(uv)
    end

    export_vtk("dissipation", dom; uv = uv)
end
