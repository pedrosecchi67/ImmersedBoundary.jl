import ImmersedBoundary.Mesher as mshr
import ImmersedBoundary as ibm

using CUDA

stl = mshr.Stereolitography("n0012.dat")

L = 20.0

meshes = mshr.Multigrid(
    5, 
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

msh = meshes[1]
dmn = ibm.Domain(msh)
dmn_coarse = ibm.Domain(meshes[end])

intp = ibm.Interpolator(dmn, dmn_coarse)

dmn = ibm.to_backend(dmn, CuArray)
intp = ibm.to_backend(intp, CuArray)

x, y = eachrow(dmn.centers)

u = y .+ x .^ 2
uavg = copy(u)
for i = 1:4
    uavg .= ibm.smooth(uavg, dmn)
end

ucoarse = intp(u)

UV = similar(dmn.centers)
UV[1, :] .= 1.0
UV[2, :] .= 0.0

ibm.impose_bc!(dmn, "wall", UV) do bdry, UVi
    ui, vi = eachrow(UVi)

    nx, ny = eachrow(bdry.normals)

    un = @. ui * nx + vi * ny

    vcat(
        (ui .- un .* nx)',
        (vi .- un .* ny)'
    )
end

dmn = ibm.to_backend(dmn, Array)
intp = ibm.to_backend(intp, Array)

u = ibm.to_backend(u, Array)
uavg = ibm.to_backend(uavg, Array)
ucoarse = ibm.to_backend(ucoarse, Array)
UV = ibm.to_backend(UV, Array)

vtk = mshr.vtk_grid("n0012", msh; uavg = uavg, u = u, UV = UV)
mshr.vtk_save(vtk)

vtk = mshr.vtk_grid("n0012_coarse", meshes[end]; u = ucoarse)
mshr.vtk_save(vtk)
