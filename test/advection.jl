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

dom = Domain(
    [0.0, 0.0],
    [1.0, 1.0],
    ("lower", lower, 1e-2),
    ("upper", upper, 1e-2);
    refinement_regions = [
        (Line([0.0, 0.0], [1.0, 1.0]), 1e-2),
    ],
    verbose = true,
)

u = zeros(length(dom))

Cx = 1.0
Cy = 1.0

# timescale for BC imposition
τ = dom() do part
    minimum(
        part.spacing ./ [Cx Cy]
    ) / 2
end |> minimum
τ *= 0.5 # CFL

Δt = 0.2

march! = (u, dt) -> begin
    dom(u) do part, u
        u .+= - (
            ∇(part, u, 1) .* Cx .+
            ∇(part, u, 2) .* Cy
        ) .* dt

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
udot = u -> let unew = copy(u)
    march!(unew, τ)

    @. (unew - u) / τ
end

mgrid = Multigrid(dom, 4)

march_implicit! = u -> begin
    r = du -> udot(u .+ du) .* Δt .- du

    du = zeros(length(u))
    A, b, prec = linearize(r, du;
        n_hutchinson_samples = 30)

    ds, _ = solve(A, b, prec;
        n_iter = 100, rtol = 1e-1,
        verbose = true,
        multigrid = mgrid)

    u .+= ds
end

for _ = 1:10
    march_implicit!(u)
end

export_vtk(
    "advection", dom;
    export_surface = false,    
    u = u,
)

end
