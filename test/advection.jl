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

dom, coarse_doms, coarseners, prolongators = MultigridDomain(
    4,
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
timescale = dom -> dom() do part
    minimum(
        part.spacing ./ [Cx Cy]
    ) / 2
end |> minimum

udot = (dom, u, dt) -> let unew = copy(u)
    dom(u, unew) do part, u, unew
        unew .+= - (
            ∇(part, u, 1) .* Cx .+
            ∇(part, u, 2) .* Cy
        ) .* dt

        impose_bc!(part, "upper", unew) do bdry, unew
            1.0f0
        end

        impose_bc!(part, "lower", unew) do bdry, unew
            0.0f0
        end
    end

    (unew .- u) ./ dt
end

mgrid_correction = (
    dom, u;
    source = 0.0f0,
    coarse_doms = [],
    coarseners = [],
    prolongators = [],
    CFL::Real = 0.5,
    n_iter::Int = 10,
) -> let uold = copy(u)
    τ = timescale(dom) * CFL

    if length(coarse_doms) > 0
        cdom = coarse_doms[1]
        coars = coarseners[1]
        prolong = prolongators[1]

        r = udot(dom, u, τ) .+ source

        uc = coars(u)
        rc = coars(r)

        τc = timescale(cdom) * CFL
        P = rc .- udot(cdom, uc, τc)

        u .+= mgrid_correction(cdom, uc;
            source = P,
            coarse_doms = coarse_doms[2:end],
            prolongators = prolongators[2:end],
            coarseners = coarseners[2:end],
            CFL = CFL, n_iter = n_iter
        ) |> prolong
    end

    for _ = 1:n_iter
        u .+= (
            source .+ udot(dom, u, τ)
        ) .* τ
    end
    s = u .- uold
    u .= uold

    s
end

for _ = 1:5
    u .+= mgrid_correction(dom, u;
        coarseners = coarseners,
        prolongators = prolongators,
        coarse_doms = coarse_doms, 
        n_iter = 25)
end

export_vtk(
    "advection", dom;
    export_surface = false,    
    u = u,
)

end
