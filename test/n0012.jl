@info "Running NACA-0012 mesh and operator test..."

import ImmersedBoundary as ibm

stl = ibm.Stereolitography("n0012.dat")

L = 20.0

dom = ibm.Domain(
    [-L/2, -L/2], [L, L],
    ("wall", stl, 0.001);
    refinement_regions = [
        ibm.Ball([0.0, 0.0], 0.0) => 0.00025,
        ibm.Ball([1.0, 0.0], 0.0) => 0.00025,
    ]
)

@info "$(length(dom)) non-blanked, non-margin cells"

dom = ibm.save_domain("dom.ibm", dom)
dom = ibm.load_domain("dom.ibm")

uv = zeros(length(dom), 2)
uv[:, 1] .= 1.0
uvcoarse = copy(uv)
k = zeros(length(dom))

dom(uv, uvcoarse, k) do part, uvdom, uvcoarse_dom, kdom

    UV = part(uvdom)
    UVc = part(uvcoarse_dom)
    K = part(kdom)

    ibm.impose_bc!(part, "wall", UV, K) do bdry, uvi, ki
        u, v = eachcol(uvi)
        nx, ny = eachcol(bdry.normals)

        uvn = @. u * nx + v * ny

        kb = similar(ki)
        kb .= 1.0

        (
            uvi .- uvn .* bdry.normals, kb
        )
    end

    ibm.impose_bc!(part, "FARFIELD", K) do bdry, ki
        kb = similar(ki)
        kb .= 0.0

        kb
    end

    UVc .= ibm.block_average(part, UV)

    ibm.update_partition!(part, uvdom, UV)
    ibm.update_partition!(part, uvcoarse_dom, UVc)
    ibm.update_partition!(part, kdom, K)

end

ibm.export_vtk("n0012_results", dom;
    uv = uv, uvcoarse = uvcoarse, k = k)
