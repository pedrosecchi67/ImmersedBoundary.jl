# ImmersedBoundary.jl

A comprehensive module for Immersed Boundary Method implementations in Julia.

## Installation

```julia
using Pkg
Pkg.add("https://github.com/pedrosecchi67/ImmersedBoundary.jl.git")
```

## Usage

Basic usage instructions are included below. **Please refer to the docstrings of each function for additional arguments and definitions**.

The example `test/cuda_rae2822.jl` shows the use of this library to obtain a GPU-parallel, 2D Euler solver for the compressible flow around an airfoil.

For a more in-depth technical explanation of the package, please refer to docs/theory.pdf.

### Stereolitography objects

Solid boundaries are described by Stereolitography input:

```julia
import ImmersedBoundary as ibm

sphere = ibm.Stereolitography("sphere.stl")

# Selig format .dat file with no header:
airfoil = ibm.Stereolitography("rae2822.dat")

circle = let theta = LinRange(0.0, 2pi, 100) |> collect
    points = [
        cos.(theta)';
        sin.(theta)'
    ]

    Stereolitography(points; closed = true)
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

    Stereolitography(points, indices)
end

# concatenate two STLs:
stl = cat(circle, airfoil)
```

### Mesh generation

Octree mesh generation can be done with:

```julia
mesh = ibm.Mesh(
    [-10.0,-10.0], [20.0,20.0], # hypercube origin, widths
    sphere => 0.01, # STL surface - max. size pair
    sphere2 => 0.005;
    clipping_surface = cat(sphere, sphere2), # clip domain at surf.
    interior_point = [5.0, 0.0], # interior point reference for domain
    n_recursive_split = 5, # number of recursive splits after boundary refinement
    refinement_regions = [
        ibm.Ball([0.0, 0.0], 0.1) => 0.005,
        ibm.Box([0.0, 0.0], [0.5, 0.5]) => 0.005,
        ibm.Line([0.0, 1.0], [1.0, 0.0]) => 0.005
    ] # pairs between distance functions and local max. sizes
)

# save to file:
vtk = ibm.vtk_grid("mesh", msh; u = rand(length(msh))) # (kwargs as cell data)
ibm.vtk_save(vtk)

# mesh size:
@info "$(ndims(msh))-dimensional mesh of size $(length(msh))"
```

### Boundary definition

To define a boundary from stereolitography objects, one may use:

```julia
bdry = ibm.Boundary(msh, stl1, stl2)
```

Or, similarly, for hypercube boundaries:

```julia
inlet = Boundary(
    msh,
    (1, false), # "back" face, first (x) axis
    (2, false), # "bottom" face, second (y) axis
    (2, true), # "top" face, second (y) axis
    (3, false), # ...
    (3, true)
)
```

### Triangle and Barnes-Hut trees

We may build a tree structure to optimize point-in-poly queries and distance calculations in the functions above.
This is done with:

```julia
tree = ibm.STLTree(stl)
```

To optimize even further, we may build an approximate distance field via a Barnes-Hut tree:

```julia
origin = [-2.0, -2.0, -2.0]
widths = [4.0, 4.0, 4.0] # hypercube boundaries for distance field range
field = ibm.DistanceField(tree, origin, widths; atol = 1e-3) # or ibm.DistanceField(stl, origin, widths; atol = 1e-3)
```

This is highly recommended for good performance with 3D geometries.

It suffices to replace the stereolitography objects by the distance fields/trees in the functions above. Just note that interior_references must be specified during `DistanceField` struct construction:

```julia
field = ibm.DistanceField(stl, origin, widths; atol = 1e-3, 
    outside_reference = [0.0, 0.0, 0.0])
```

...and other definitions will be void.

### Residual evaluation

To evaluate residuals, one may use stencil points and cell spacing information for the mesh.

For example, with a vector of cell properties $u$, one may find $\partial u/\partial x$ using central differences:

```julia
u = rand(length(msh))

dx, dy = msh.spacing

ux = (
    msh(u, 1, 0) .- msh(u, -1, 0)
) ./ (2 .* dx)

# similarly:
uy = (
    getalong(u, msh, 2, 1) .- # vector, mesh, dimension, offset
    getalong(u, msh, 2, -1)
) ./ (2 .* dy)
```

These functions also work for multidimensional tensors, so long as the last dimension indicates the cell index.

### Boundary condition imposition

To impose (in-place) the non-penetration condition for a velocity field at boundary `bdry`, one may use:

```julia
bc = ibm.BoundaryCondition() do bdry, u, v # field variables interpolated to image points before imposition
    nx, ny = bdry.normals

    unormal = @. nx * u + ny * v

    ( # note that fewer return values than input field variables may be specified, if needed
        u .- nx .* unormal,
        v .- ny .* unormal
    )
end

ibm.impose_bc!(bc, bdry, u, v)
```

Check `?ibm.Boundary` and `?ibm.impose_bc!` for further information.

### Surfaces and postprocessing

In this package, the term "surface" is used to identify any triangulated surface for postprocessing purposes, while "boundary" refers to a limit of the numerical domain on which the user may impose boundary conditions.

To create a surface, one may use:

```julia
surf = ibm.Surface(
    msh, stl;
    linear = true # use linear interpolation
)

# to refine via tri splitting to reach a given maximum surface element length:
surf = ibm.Surface(
    msh, stl, max_length
)
```

One may interpolate fluid properties to a surface and use them for integration:

```julia
# "velocity" vector in 2D mesh:
uv = rand(2, length(msh))

uv_surf = surf(uv)
Cp = let (u, v) = eachrow(uv_surf)
    @. 1.0 - u ^ 2 - v ^ 2
end

nx, ny = surf.normals # check docstring for other properties

CX = ibm.surface_integral(surf, - nx .* Cp)
CY = ibm.surface_integral(surf, - ny .* Cp)
```

Surfaces may also be used to generate VTK output:

```julia
vtk = ibm.surf2vtk("surf_output", surf; u = u, v = v) # kwargs as node data
ibm.vtk_save(vtk)
```

### Multigrid

Geometric multigrid coarsening may be defined with:

```julia
mgrid = ibm.Multigrid(msh)

# for max. block size 2 ^ (ndims(msh) * 4):
mgrid = ibm.Multigrid(msh, 4)

Q = rand(4, length(msh))

Qcoarse = mgrid(Q)

# ratio between coarse and fine cell sizes:
@show mgrid.size_ratios
```

Check the docstrings for further info.

### GPU/CPU parallelization

Function `to_backend` may be used to convert all arrays to a custom backend, such as that of CUDA.jl:

```julia
##################################################
# port everything to GPU

# Mesh:
msh = ibm.to_backend(CuArray, msh)
# Multigrid:
mgrid_levels = map(mgrid -> ibm.to_backend(CuArray, mgrid), mgrid_levels)
# Boundaries:
wall = ibm.to_backend(CuArray, wall)
freestream = ibm.to_backend(CuArray, freestream)

# State variables:
Q = CuArray(Q)

##################################################

# Run calcs.

##################################################
# port everything to CPU

msh = ibm.to_backend(Array, msh)
mgrid_levels = map(mgrid -> ibm.to_backend(Array, mgrid), mgrid_levels)
wall = ibm.to_backend(Array, wall)
freestream = ibm.to_backend(Array, freestream)

Q = Array(Q)

##################################################
```

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
