import ImmersedBoundary.Mesher as mshr
import ImmersedBoundary as ibm

stl = mshr.Stereolitography("n0012.dat")

L = 20.0

msh = mshr.FixedMesh(
    [-L/2,-L/2], [L,L],
    ("wall", stl, 0.001);
    verbose = true,
    farfield_boundaries = [
        "farfield" => [(1, false), (1, true), (2, false), (2, true)]
    ],
    refinement_regions = [
        mshr.Ball([0.0, 0.0], 0.01) => 0.00025,
        mshr.Ball([1.0, 0.0], 0.01) => 0.00025,
    ]
)

dmn = ibm.Domain(msh)

x, y = eachrow(msh.centers)

u = y .+ x .* 2

ux = ibm.∇(u, dmn, 1)
uy = ibm.∇(u, dmn, 2)

vtk = mshr.vtk_grid("n0012", msh; ux = ux, uy = uy)
mshr.vtk_save(vtk)
