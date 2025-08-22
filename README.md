# ImmersedBoundary.jl

A comprehensive module for Immersed Boundary Method implementations in Julia
and their parallelization via SIMD GPU computing.

## Installation

```julia
using Pkg
Pkg.add("https://github.com/pedrosecchi67/ImmersedBoundary.jl.git")
```

## Usage

Basic usage instructions are included below. **Please refer to the docstrings of each function for additional arguments and definitions**.

For a more in-depth theoretical explanation of the package, please refer to docs/theory.pdf.

**Note that all functions below work in both 2D and 3D.**

### Stereolitography objects

Stereolitography objects can be used to describe surfaces:

```julia
import ImmersedBoundary as ibm

# binary or ASCII:
sphere = ibm.Stereolitography("sphere.stl")

# Selig format .dat file with no header:
airfoil = ibm.Stereolitography("rae2822.dat")

circle = let theta = LinRange(0.0, 2pi, 100) |> collect
    points = [
        cos.(theta)';
        sin.(theta)'
    ]

    ibm.Stereolitography(points; closed = true)
end

# same, but with defined simplex corner indices:
circle = let theta = LinRange(0.0, 2pi, 100) |> collect
    points = [
        cos.(theta[1:(end - 1)])';
        sin.(theta[1:(end - 1)])'
    ]
    indices = [
        collect(1:length(theta))'
        circshift(collect(1:length(theta)), -1)'
    ]

    ibm.Stereolitography(points, indices)
end

# concatenate two STLs:
stl = cat(circle, airfoil)
```

### Mesh/domain definition

```julia
function Domain(
        origin::Vector{Float64}, widths::Vector{Float64},
        surfaces::Tuple{String, Stereolitography, Float64}...;
        refinement_regions::AbstractVector = [],
        max_length::Float64 = Inf,
        growth_ratio::Float64 = 1.1,
        ghost_layer_ratio::Tuple = (-1.1, 2.1),
        interior_point = nothing,
        approximation_ratio::Float64 = 2.0,
        verbose::Bool = false,
        max_partition_cells::Int64 = 1000_000,
        families = nothing,
        stencil_points = Tuple[],
)
```

Generate an octree mesh defined by:

* A hypercube origin;
* A vector of hypercube widths;
* A set of tuples in format `(name, surface, max_length)` describing
    stereolitography surfaces (`Mesher.Stereolitography`) and 
    the max. cell widths at these surfaces;
* A set of refinement regions described by distance functions and
    the local refinement at each region. Example:
        ```julia
        refinement_regions = [
            ibm.Ball([0.0, 0.0], 0.1) => 0.005,
            ibm.Ball([1.0, 0.0], 0.1) => 0.005,
            ibm.Box([-1.0, -1.0], [3.0, 2.0]) => 0.0025,
            ibm.Line([1.0, 0.0], [2.0, 0.0]) => 0.005,
            ibm.Triangulation(
                ibm.feature_edges(stl;
                    angle = 10.0, radius = 0.05) # detects feature edges in solids
            ) => 0.001
        ]
        ```
* A maximum cell size (optional);
* A volumetric growth ratio;
* An interval of ratios between the cell circumradius and the SDF of a given surface for 
    which cells are defined as ghosts;
* A point reference within the domain. If absent, external flow is assumed;
* An approximation ratio between wall distance and cell circumradius past which
    distance functions are approximated; 
* A maximum number of cells per partition;
* A set of families defining surface groups for postprocessing, BC imposition and wall
    distance calculations.

The families may be defined with the following syntax:

```julia
surfaces = [
    ("wing", "wing.stl", 1e-3),
    ("flap", "flap.stl", 1e-3)
]
families = [
    "wall" => ["wing", "flap"], # a group of surface names
    "inlet" => [
        (1, false), # fwd face, first dimension (x)
        (2, false), # left face, second dimension (y)
        (2, true), # right face, second dimension (y)
        (3, false), # bottom face, third dimension (z)
        (3, true), # top face, third dimension (z)
    ],
    "outlet" => [(1, true)]
]

dom = ibm.Domain(origin, widths, surfaces...; families = families)
```

The default family definition uses each surface as a family and defines
the farfield as `"FARFIELD"`.

Stencil points may be specified as tuples. An example for first order operations
on a 3D mesh is:

```julia
stencil_points = [
    (-1, 0, 0), (0, 0, 0), (1, 0, 0),
    (0, -1, 0), (0, 0, 0), (0, 1, 0), # don't worry about duplicate values
    (0, 0, -1), (0, 0, 0), (0, 0, 1)
]
```

The default is a cruciform, second-order CFD stencil.

Meshes can be saved to binary (serialized) files:

```julia
ibm.save_domain("domain.ibm", dom)
dom = ibm.load_domain("domain.ibm")
```

Or used to build VTK output:

```julia
u = rand(length(dom)) 
v = rand(length(dom), 2) 

ibm.export_vtk("results", dom; 
    include_volume = true, include_surface = true,
    u = u, v = v)
```

Arrays can be passed as kwargs to record cell data. The first dimension is assumed to refer to the cell index.

### Partitioned residual computing

ImmersedBoundary.jl implements mesh paritioning to ensure that only a bunch of blocks in the building-cubes mesh has its residuals evaluated concurrently, thus saving on the amount of RAM used at any given time. This also allows for easier GPU parallelization, which involves tighter memory limits.

The following example is given:

```julia
domain(r, u) do partition, rdom, udom
    # udom includes the parts of array `u`
    # which affect the residual at partition `partition`.

    # now do some Cartesian grid operations and
    # update rdom in-place
end

# after the loop, the values of `rdom` are returned to
# array `r`, in-place.
```

It's also an option to automatically convert the passed arrays and partition info to a given array backend (e.g. `CuArray`) to perform the residual calculations:

```julia
dom(
    u;
    conv_to_backend = x -> CuArray(x),
    conv_from_backend = x -> Array(x)
) do part, udom
    @show typeof(udom) # CuArray
end
```

The return values of each function call are also gathered and returned in a vector.

### PDE discretization

Cartesian grid operations can be done with `(part::Partition)()` and `ibm.getalong` acting over partition arrays:

```julia

u = rand(length(domain))
ux = similar(u)

domain(u, ux) do part, u, ux # u, ux for local domain cells only
    # in a 3D mesh:
    dx, dy, dz = part.spacing |> eachcol

    ux .= (
        part(u, 1, 0, 0) .- part(u, -1, 0, 0)
    ) ./ (2 .* dx)

    # or simply:
    ux .= ibm.δ(part, u, 1)
end

# ux is now the first, x-axis derivative of u
```

Other functions include:

```julia
ibm.∇ # backward derivative
ibm.Δ # forward derivative
ibm.δ # central derivative
ibm.μ # face averages
ibm.MUSCL # MUSCL reconstruction
```

### Boundary conditions

BC imposition is performed via ghost cells (check the documentation for further info), each having an image point on the opposite side of the boundary. An example of BC imposition is the following implementation of the non-penetration condition:

```julia
dom(u, v) do part, u, v
    ibm.impose_bc!(part, "wall", u, v) do bdry, uimage, vimage
        nx, ny = bdry.normals |> eachcol
        un = @. nx * uimage + ny * vimage

        ( # returns values at boundary given values at image point
            uimage .- un .* nx,
            vimage .- vn .* ny
        )
    end
end


# alternative return value:
uv = zeros(length(dom), 2)
uv[:, 1] .= 1.0
dom(uv) do part, uv
    ibm.impose_bc!(part, "wall", uv) do bdry, uvim
        uimage, vimage = eachcol(uvim)
        nx, ny = eachcol(bdry.normals)
        un = @. nx * uimage + ny * vimage

        uvim .- un .* bdry.normals
    end
end
```

The `Boundary` type struct also includes the following fields:

```julia
bdry.points # shape (nghosts, ndim): boundary projections of ghost cells
bdry.normals # shape (nghosts, ndim)
bdry.image_distances # to wall
bdry.ghost_distances # to wall
```

Note that other field variable arrays may be passed to `impose_bc!`, and only the first input arrays will have their values altered as per the returned boundary values. Example:

```julia
dom(u, v) do part, u, v
    ibm.impose_bc!(part, "boundary", u, v, w) do bdry, ui, vi, wi
        # Neumann, du!dn = 2v
        ubdry = ui .- vi .* 2.0 .* bdry.image_distances
        vbdry = vi .+ wi # arbitrary

        (ubdry, vbdry)
    end
end
```

### Wall distances

Wall distances for family `"wall"` may be accessed with:

```julia
signed_distance = dom.boundary_distances["wall"]
```

### Surfaces and postprocessing

Surfaces may be used for postprocessing and coefficient integration. Example:

```julia
surf = dom.surfaces["wall"]

Cp_wall = surf(Cp) # interpolate array of field properties to wall

CX, CY = ibm.surface_integral(
    surf, Cp_wall .* surf.normals
)
```

### CFD utilities

For easier implementation of CFD codes, you may use the module `ImmersedBoundary.CFD`. Check the docstrings for the following functions:

```julia
?ibm.MUSCL
?ibm.CFD.Fluid
?ibm.CFD.speed_of_sound
?ibm.CFD.heat_conductivity
?ibm.CFD.dynamic_viscosity
?ibm.CFD.viscous_fluxes
?ibm.CFD.state2primitive
?ibm.CFD.primitive2state
?ibm.CFD.rms
?ibm.CFD.HLL
?ibm.CFD.AUSM
?ibm.CFD.JSTKE
?ibm.CFD.pressure_coefficient
?ibm.CFD.Freestream
?ibm.CFD.initial_guess
?ibm.CFD.rotate_and_rescale!
?CFD.timescale
?CFD.wall_bc
?CFD.freestream_bc
?CFD.ConvergenceCriteria
?CFD.CTUCounter
?CFD.clip_CFL!
```
