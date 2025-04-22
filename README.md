# ImmersedBoundary.jl

A comprehensive module for Immersed Boundary Method implementations in Julia.

## Installation

```julia
using Pkg
Pkg.add("https://github.com/pedrosecchi67/ImmersedBoundary.jl.git")
```

## Usage

Basic usage instructions are included below. **Please refer to the docstrings of each function for additional arguments and definitions**.

For a more in-depth technical explanation of the package, please refer to docs/theory.pdf.

### Stereolitography objects

Solid boundaries are described by Stereolitography input:

```julia
import ImmersedBoundary as ibm
import ImmersedBoundary.Mesher as mshr

sphere = mshr.Stereolitography("sphere.stl")

# Selig format .dat file with no header:
airfoil = mshr.Stereolitography("rae2822.dat")

circle = let theta = LinRange(0.0, 2pi, 100) |> collect
    points = [
        cos.(theta)';
        sin.(theta)'
    ]

    mshr.Stereolitography(points; closed = true)
end

circle = let theta = LinRange(0.0, 2pi, 100) |> collect
    points = [
        cos.(theta[1:(end - 1)])';
        sin.(theta[1:(end - 1)])'
    ]
    indices = [
        collect(1:length(theta))'
        circshift(collect(1:length(theta)), -1)'
    ]

    mshr.Stereolitography(points, indices)
end

# concatenate two STLs:
stl = cat(circle, airfoil)
```

### Mesh generation

Fixed octree mesh generation can be done with:

```julia
    function FixedMesh(
            origin::Vector{Float64}, widths::Vector{Float64},
            surfaces::Tuple{String, Stereolitography, Float64}...;
            refinement_regions::AbstractVector = [],
            growth_ratio::Float64 = 1.2,
            max_length::Float64 = Inf,
            ghost_layer_ratio::Float64 = -2.2,
            interior_point = nothing,
            approximation_ratio::Float64 = 2.0,
            filter_triangles_every::Int64 = 0,
            verbose::Bool = false,
            farfield_boundaries = nothing,
    )
```

* A hypercube origin;
* A vector of hypercube widths;
* A set of tuples in format `(name, surface, max_length)` describing
    stereolitography surfaces (`Mesher.Stereolitography`) and 
    the max. cell widths at these surfaces;
* A set of refinement regions described by distance functions and
    the local refinement at each region. Example:

```julia
refinement_regions = [
    Mesher.Ball([0.0, 0.0], 0.1) => 0.005,
    Mesher.Ball([1.0, 0.0], 0.1) => 0.005,
    Mesher.Box([-1.0, -1.0], [3.0, 2.0]) => 0.0025,
    Mesher.Line([1.0, 0.0], [2.0, 0.0]) => 0.005
]
```

* A cell growth ratio;
* A maximum cell size (optional);
* A ratio between the cell circumradius and the SDF threshold past which
    cells are considered to be out of the domain. `ghost_layer_ratio = -2.0` 
    guarantees that a layer of at least two ghost cell layers are included 
    within each solid;
* A point reference within the domain. If absent, external flow is assumed;
* An approximation ratio between wall distance and cell circumradius past which
    distance functions are approximated;
* A number of recursive refinement levels past which the triangles in the provided
    triangulations are filtered to lighter, local topologies.

Farfield boundaries may be defined with the following syntax:

```julia
farfield_boundaries = [
    "inlet" => [
        (1, false), # fwd face, first dimension (x)
        (2, false), # left face, second dimension (y)
        (2, true), # right face, second dimension (y)
        (3, false), # bottom face, third dimension (z)
        (3, true), # top face, third dimension (z)
    ],
    "outlet" => [(1, true)]
]
```

Meshes can be saved to JSON format files:

```julia
mshr.mesh2json("mesh.json", msh)
msh = mshr.json2mesh("mesh.json")
```

Or used to build VTK output:

```julia
u = rand(length(msh)) 
v = rand(2, length(msh)) 

vtk = mshr.mesh2vtk("results", msh; u = u, v = v)
mshr.vtk_save(vtk)
```

Arrays can be passed as kwargs to record cell data. The last dimension is assumed to refer to the cell index.

### Domain definition

To define a PDE domain from a mesh, you can simply use:

```julia
domain = ibm.Domain(msh)
```

Check out the docstring for tunable hyperparameters regarding ghost points.

### Residual evaluation

To evaluate residuals, one may use stencil points and cell spacing information for the mesh.

For example, with a vector of cell properties $u$, one may find $\partial u/\partial y$ using central differences:

```julia
dx, dy = eachrow(
    domain.widths
)

du!dy = (
    domain(u, 0, 1) .- domain(u, 0, -1)
) ./ (2 .* dy)
```

These functions also work for multidimensional arrays, so long as the last dimension indicates the cell index.

### Boundary condition imposition

To impose (in-place) the non-penetration condition for a velocity field at boundary `bdry`, one may use:

```julia
impose_bc!(domain, "wall", u, v) do bdry, ui, vi # boundary struct and values at image points
    nx, ny = eachrow(bdry.normals)

    un = @. ui * nx + vi * ny

    (
        ui .- un .* nx,
        vi .- un .* ny
    )
end
```

Check `?ibm.Boundary` and `?ibm.impose_bc!` for further information.

### Surfaces and postprocessing

In this package, the term "surface" is used to identify any triangulated surface for postprocessing purposes, while "boundary" refers to a limit of the numerical domain on which the user may impose boundary conditions.

To create a surface, one may use:

```julia
surf = ibm.Surface(
    domain, stl
)

# to refine via tri splitting to reach a given maximum surface element length:
surf = ibm.Surface(
    domain, stl, max_length
)

# to use a stereolitography object previously employed for boundary definition:
surf = ibm.Surface(domain, "wall")
```

One may interpolate fluid properties to a surface and use them for integration:

```julia
# "velocity" vector in 2D mesh:
uv = rand(2, length(domain.mesh))

uv_surf = surf(uv)
Cp = let (u, v) = eachrow(uv_surf)
    @. 1.0 - u ^ 2 - v ^ 2
end

nx, ny = surf.normals |> eachrow # check docstring for other properties

CX = ibm.surface_integral(surf, - nx .* Cp)
CY = ibm.surface_integral(surf, - ny .* Cp)
```

Surfaces may also be used to generate VTK output:

```julia
vtk = ibm.surf2vtk("surf_output", surf; u = u, v = v) # kwargs as node data
ibm.vtk_save(vtk)
```

### Multigrid

To generate a series of meshes with element size ratios of 2 near the boundary:

```julia
m1, m2, m3 = mshr.Multigrid( # from finest to coarsest
    3, # 3 levels
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
```

To interpolate flow properties between domains:

```julia
dmn1 = ibm.Domain(m1)
dmn2 = ibm.Domain(m2)

intp = ibm.Interpolator(dmn1, dmn2)

Q1 = rand(length(dmn1.mesh))
Q2 = intp(Q1)
```

### GPU/CPU parallelization

Function `to_backend` may be used to convert all arrays to a custom backend, such as that of CUDA.jl:

```julia
dmn = ibm.to_backend(dmn, CuArray) # domains
surf = ibm.to_backend(surf, CuArray) # surfaces
intp = ibm.to_backend(intp, CuArray) # interpolators

# ...and back:
dmn = ibm.to_backend(dmn, Array)
```

### Batch residual evaluation

GPUs (and cheap computers!) have tight memory limits, which may make it tricky to evaluate residuals in large meshes.

To mitigate this problem, we provide the `BatchResidual` struct:

```julia
residual = ibm.BatchResidual(
    dmn; 
    max_size = 10000
) do domain, Q
    domain = ibm.to_backend(domain, CuArray)
    Q = CuArray(Q)

    u, v = eachrow(Q)

    ibm.impose_bc!(domain, "wall", u, v) do bdry, U, V
        nx, ny = eachrow(bdry.normals)
        un = @. U * nx + V * ny

        (
            U .- un .* nx,
            V .- un .* ny
        )
    end

    [u'; v'] |> Array
end

R = residual(Q)
```

Which creates mesh partitions of, at most, `max_size` cells and calculates the residual at one partition at a time.

Note that this involves transporting small batches of data to and from a GPU, which may be costly with a small `max_size`.

### CFD utilities

For easier implementation of CFD codes, you may use the module `ImmersedBoundary.CFD`. Check the docstrings for the following functions:

```julia
?ibm.MUSCL
?ibm.CFD.Fluid
?ibm.CFD.speed_of_sound
?ibm.CFD.state2primitive
?ibm.CFD.primitive2state
?ibm.CFD.rms
?ibm.CFD.HLL
?ibm.JSTKE
```
