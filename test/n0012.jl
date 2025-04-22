import ImmersedBoundary.Mesher as mshr
import ImmersedBoundary as ibm

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

x, y = eachrow(dmn.centers)

u = y .+ x .^ 2
uavg = copy(u)
for i = 1:4
    uavg .= ibm.smooth(uavg, dmn)
end

residual = ibm.Residual(dmn; max_size = 10000) do domain, Q
    u, v = copy(Q) |> eachrow

    ibm.impose_bc!(domain, "wall", u, v) do bdry, U, V
        nx, ny = eachrow(bdry.normals)
        un = @. U * nx + V * ny

        (
            U .- un .* nx,
            V .- un .* ny
        )
    end

    [u'; v']
end

Q = zeros(2, length(msh))
Q[1, :] .= 1.0
R = residual(Q)

vtk = mshr.vtk_grid("n0012", msh; uavg = uavg, u = u, Q = Q, R = R)
mshr.vtk_save(vtk)

dmn_coarse = ibm.Domain(meshes[end])
intp = ibm.Interpolator(dmn, dmn_coarse)
u = intp(u)

vtk = mshr.vtk_grid("n0012_coarse", meshes[end]; u = u)
mshr.vtk_save(vtk)

sub = residual.subdomains[15]
vtk = mshr.vtk_grid("n0012_subdomain", sub.mesh)
mshr.vtk_save(vtk)
