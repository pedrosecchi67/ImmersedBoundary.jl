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

A building-cubes mesh may be defined with:

```julia
function Domain(
        origin::Vector{Float64}, widths::Vector{Float64},
        surfaces::Tuple{String, Stereolitography, Float64}...;
        margin::Int64 = 2,
        block_sizes::Union{Tuple, Int} = 8,
        refinement_regions::AbstractVector = [],
        max_length::Float64 = Inf,
        ghost_layer_ratio::Tuple = (-1.1, 2.1),
        interior_point = nothing,
        approximation_ratio::Float64 = 2.0,
        verbose::Bool = false,
        max_partition_blocks::Int64 = 1000,
        multigrid_levels::Int64 = 0,
        families = nothing,
)
```

* A hypercube origin;
* A vector of hypercube widths;
* A set of tuples in format `(name, surface, max_length)` describing
    stereolitography surfaces (`Mesher.Stereolitography`) and 
    the max. cell widths at these surfaces;
* A "margin" of cells for each block, used for inter-block communication;
* A block size, in the form of a tuple with number of cells along each axis, or
    an integer with the same number for all axes;
* A set of refinement regions described by distance functions and
    the local refinement at each region. Example:
        ```julia
        refinement_regions = [
            ibm.Ball([0.0, 0.0], 0.1) => 0.005,
            ibm.Ball([1.0, 0.0], 0.1) => 0.005,
            ibm.Box([-1.0, -1.0], [3.0, 2.0]) => 0.0025,
            ibm.Line([1.0, 0.0], [2.0, 0.0]) => 0.005
        ]
        ```
* A maximum cell size (optional);
* An interval of ratios between the cell circumradius and the SDF of a given surface for 
    which cells are defined as ghosts;
* A point reference within the domain. If absent, external flow is assumed;
* An approximation ratio between wall distance and cell circumradius past which
    distance functions are approximated; 
* A maximum number of octree blocks per partition; 
* A number of multigrid levels; and
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

    U = part(udom) # obtain Cartesian representation
    # of u. Shape (nblocks, nx, ny[, nz])
    R = part(rdom)

    # now do some Cartesian grid operations and
    # update R

    ibm.update_partition!(part, rdom, R) # send values to `rdom`
end

# after the loop, the values of `rdom` are returned to
# array `r`
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

Cartesian grid operations can be done with `(part::Partition)()` and `ibm.getalong` acting over block-structured arrays:

```julia

u = rand(length(domain))
ux = similar(u)

domain(u, ux) do part, udom, uxdom
    U = part(udom)
    Ux = part(uxdom)

    dx, dy = part.spacing |> eachcol

    Ux .= (
        part(U, 1, 0, 0) .- part(U, -1, 0, 0)
    ) ./ (2 .* dx)

    # equivalent to part(U, 0, 0, 2):
    Ukp2 = ibm.getalong(part, U, 3, 2)

    ibm.update_partition!(part, uxdom, Ux)
end

# ux is now the first, x-axis derivative of u
```

### Boundary conditions

BC imposition is performed via ghost cells (check the documentation for further info), each having an image point on the opposite side of the boundary. An example of BC imposition is the following implementation of the non-penetration condition:

```julia
dom(u, v) do part, udom, vdom
    U = part(udom)
    V = part(vdom)

    ibm.impose_bc!(part, "wall", U, V) do bdry, uimage, vimage
        nx, ny = bdry.normals |> eachcol
        un = @. nx * uimage + ny * vimage

        ( # returns values at boundary given values at image point
            uimage .- un .* nx,
            vimage .- vn .* ny
        )
    end

    ibm.update_partition!(part, udom, U)
    ibm.update_partition!(part, vdom, V)
end


# alternative return value:
uv = zeros(length(dom), 2)
uv[:, 1] .= 1.0
dom(uv) do part, uvdom
    UV = part(uvdom)
    
    ibm.impose_bc!(part, "wall", UV) do bdry, uvim
        uimage, vimage = eachcol(uvim)
        nx, ny = eachcol(bdry.normals)
        un = @. nx * uimage + ny * vimage

        uvim .- un .* bdry.normals
    end

    ibm.update_partition!(part, uvdom, UV)
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
dom(u, v) do part, udom, vdom
    U = part(udom)
    V = part(vdom)

    ibm.impose_bc!(part, "boundary", U, V) do bdry, ui, vi
        # Neumann, du!dn = 2v
        ubdry = ui .- vi .* 2.0 .* bdry.image_distances

        ubdry
    end

    ibm.update_partition!(part, uvom, U)
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

### Multigrid

You can use `ibm.block_average` to obtain the average of a block-structured array at each mesh block in the current partition, and thus obtain one level of agglomeration-based,
geometric multigrid:

```julia
dom(r, u) do part, rdom, udom
    U = part(udom)

    # calculate residual R somehow

    # "coarsen" by taking the average of each mesh block:
    R .= ibm.block_average(part, R; include_blank = true)

    # rescale time-step by using part.block_size
    # to obtain a coarse/fine mesh size ratio:
    R .*= (
        min(part.block_size...)
    )

    ibm.update_partition!(part, rdom, R)
end
```

Another more elaborate, but more memory-consuming alternative is the 
kwarg `multigrid_levels` in the `Domain` constructor:

```julia
function solve_multigrid(dom, Q, R)
    if isnothing(dom.multigrid) # stopping condition: coarsest level
        return solve(dom, Q, R) # solve for residual on current level
    end

    # solve on coarse level and prolongate
    coarsener, dom_coarse, prolongator = dom.multigrid
    dQ = solve(dom_coarse, coarsener(Q), coarsener(R)) |> prolongator

    dQ .+= solve(dom, Q, R) # solve for residual on current level

    dQ
end

dom = ibm.Domain(origin, widths, surfaces...;
        multigrid_levels = 3)
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
?ibm.CFD.AUSM
?ibm.CFD.JSTKE
?ibm.CFD.pressure_coefficient
?ibm.CFD.Freestream
?ibm.CFD.initial_guess
?ibm.CFD.rotate_and_rescale!
?CFD.timescale
?CFD.wall_bc
?CFD.freestream_bc
```
