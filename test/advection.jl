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
    max_partition_size = 1000,
    multigrid_levels = 4)

u = zeros(length(dom))

dt = 0.1
Cx = 1.0
Cy = 1.0

uold = copy(u)
du, _, _ = newton_rhapson(
    dom, u, uold
) do domain, u, uold
    r = similar(u)
    r .= 0.0

    domain(u, uold, r) do part, u, uold, r
        global dt, Cx, Cy

        unew = uold .- dt .* (
            ∇(part, u, 1) .* Cx .+ ∇(part, u, 2) .* Cy
        ) 

        impose_bc!(part, "lower", unew) do bdry, u
            ub = similar(u)
            ub .= 0.0
            ub
        end
        impose_bc!(part, "upper", unew) do bdry, u
            ub = similar(u)
            ub .= 1.0
            ub
        end

        @. r = unew - u
    end

    r
end
u .+= du

export_vtk(
    "advection", dom;
    include_surface = false,    
    u = u,
)

end
