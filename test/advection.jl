begin

@info "Running linear advection test case"

lower = Stereolitography(
    [
        1.0 0.0;
        0.0 0.0
    ]; closed = false
)
upper = Stereolitography(
    [
        0.0 0.0;
        0.0 1.0
    ]; closed = false
)

msh = Mesh(
    [0.0, 0.0],
    [1.0, 1.0],
    ("lower", lower, 1e-2),
    ("upper", upper, 1e-2);
    refinement_regions = [
        (Line([0.0, 0.0], [1.0, 1.0]), 1e-2),
    ],
    interior_reference = [0.5, 0.5],
    verbose = true,
)

dom = Domain(msh;
    max_partition_size = 1000)

u = zeros(length(dom))

Cx = 1.0
Cy = 1.0

Δt = dom() do part
    minimum(
        part.spacing ./ [Cx Cy]
    ) / 2
end |> minimum
Δt *= 0.5 # CFL

march! = u -> begin
    dom(u) do part, u
        u .+= - (
            ∇(part, u, 1) .* Cx .+
            ∇(part, u, 2) .* Cy
        ) .* Δt

        impose_bc!(part, "upper", u) do bdry, u
            ub = similar(u)
            ub .= 1.0

            ub
        end

        impose_bc!(part, "lower", u) do bdry, u
            ub = similar(u)
            ub .= 0.0

            ub
        end
    end
end

for _ = 1:100
    march!(u)
end

export_vtk(
    "advection", dom;
    include_surface = false,    
    u = u,
)

end
