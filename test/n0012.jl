@info "Running NACA-0012 mesh and operator test..."

import ImmersedBoundary as ibm

stl = ibm.Stereolitography("n0012.dat")

L = 20.0

dom = ibm.Domain(
    [-L/2, -L/2], [L, L],
    ("wall", stl, 0.01);
    refinement_regions = [
        ibm.Ball([0.0, 0.0], 0.0) => 0.0025,
        ibm.Ball([1.0, 0.0], 0.0) => 0.0025,
    ],
    max_partition_cells = 5000,
    verbose = true
)

@info "$(length(dom)) non-blanked, non-margin cells"

ibm.save_domain("dom.ibm", dom)
dom = ibm.load_domain("dom.ibm")

fluid = ibm.CFD.Fluid()
free_old = ibm.CFD.Freestream(fluid, 0.5, 0.0)
free = ibm.CFD.Freestream(fluid, 0.5, 2.0)

ρ, E, ρu, ρv = ibm.CFD.initial_guess(free_old, length(dom))
ibm.CFD.rotate_and_rescale!(free_old, free, ρ, E, ρu, ρv)

Q = [ρ E ρu ρv]

for _ = 1:10
    P = ibm.CFD.state2primitive(fluid, Q)
    dt = ibm.timescale(dom, fluid, P) |> minimum
    dt *= 0.5

    let Qnew = copy(Q)
        dom(Q, P, Qnew) do part, Q, P, Qnew
            for i = 1:2
                Pl, Pr = ibm.MUSCL(part, P, i)

                Qnew .-= dt .* ibm.∇(part, ibm.CFD.HLL(Pl, Pr, i, fluid), i)
            end

            Pnew = ibm.CFD.state2primitive(fluid, Qnew)

            ibm.impose_bc!(
                ibm.wall_bc,
                part, "wall",
                Pnew; fluid = fluid,
                du!dn = (b, v, p) -> let dv = similar(v)
                    dv .= 100.0
                    dv
                end
            )
            ibm.impose_bc!(
                ibm.freestream_bc,
                part, "FARFIELD",
                Pnew;
                freestream = free
            )

            Qnew .= ibm.CFD.primitive2state(fluid, Pnew)
        end

        Q .= Qnew
    end
end

ρ, E, ρu, ρv = eachcol(Q)

ibm.export_vtk("n0012_results", dom;
    rho = ρ, E = E, rhou = ρu, rhov = ρv)