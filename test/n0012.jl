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
    dt = ibm.timescale(dom, fluid, Q) |> minimum
    dt *= 0.5

    let Qnew = copy(Q)
        dom(Q, Qnew) do part, Q, Qnew
            for i = 1:2
                Ql, Qr = ibm.MUSCL(part, Q, i)

                Qnew .-= dt .* ibm.∇(part, ibm.CFD.HLL(Ql, Qr, i, fluid), i)
            end

            ibm.impose_bc!(
                ibm.wall_bc,
                part, "wall",
                Qnew; fluid = fluid,
                du!dn = (b, v, q) -> let dv = similar(v)
                    dv .= 100.0
                    dv
                end
            )
            ibm.impose_bc!(
                ibm.freestream_bc,
                part, "FARFIELD",
                Qnew;
                freestream = free
            )
        end

        Q .= Qnew
    end
end

ρ, E, ρu, ρv = eachcol(Q)

ibm.export_vtk("n0012_results", dom;
    rho = ρ, E = E, rhou = ρu, rhov = ρv)