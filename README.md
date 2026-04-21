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
sphere = Stereolitography("sphere.stl")

# Selig format .dat file with no header:
airfoil = Stereolitography("rae2822.dat")

circle = let theta = LinRange(0.0, 2pi, 100) |> collect
    points = [
        cos.(theta)';
        sin.(theta)'
    ]

    Stereolitography(points; closed = true)
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

    Stereolitography(points, indices)
end

# concatenate two STLs:
stl = cat(circle, airfoil)

# refine STL object to a given max. length via tet splitting:
stl = refine_to_length(stl, 0.001)

# merge points in one or more STL:
stl = merge_points(stl1, stl2; tolerance = 1e-5)
```

# Mesh generation

```julia
function Domain(
    origin::AbstractVector, widths::AbstractVector,
    surfaces...;
    growth_ratio::Real = 2.0f0,
    tolerance::Real = 1f-7,
    block_size::Int = 8,
    refinement_regions::AbstractVector = [],
    margin::Int = 2,
    max_partition_size::Int = 1_000_000,
    ghost_layer_ratio::Real = 1.5f0,
    hypercube_families = [],
    verbose::Bool = false,
)
```

Generate an octree/quadtree mesh described by:

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
* A "cell growth ratio" for the octree/quadtree blocks, following Nakahashi's building-cubes method; and
* A ghost layer ratio, which defines the thickness of the ghost cell layer
    within a solid as a ratio of the local cell circumdiameter.

Hypercube boundary family names may be specified by:

```julia
hypercube_families = [
    "inlet" => [
        (1, false), # back face, x axis
        (2, true), # front face, y axis
        (3, false), # bottom face, z axis
        (3, true) # top face, z axis
    ],
    "symmetry" => [
        (2, false) # left face, y axis
    ],
    "outlet" => [
        (1, true) # front face, x axis
    ]
]
```

Example:

```julia
stl = Stereolitography("rae2822.dat")
features = feature_regions(stl; radius = 0.05)
feature_dfield = DistanceField(features)

dom = Domain(
    [-40.0, -40.0],
    [80.0, 80.0],
    ("wall", stl, 4e-3);
    refinement_regions = [
        (feature_dfield, 0.5e-3),
    ],
    hypercube_families = [
        "freestream" => [(1, false), (1, true), (2, false), (2, true)],
    ],
    max_partition_size = 2000,
    verbose = true
)

@show ndims(dom) # dimensionality
@show length(dom) # number of cells
```

# Calculating residuals

```julia
function (dom::Domain)(
    f,
    args::AbstractArray...; 
    conv_to_backend = identity,
    conv_from_backend = identity,
    nthreads::Int = 1,
    kwargs...
)
```

Run function on all partitions of a domain.

Example:

```julia
domain(A, B) do part, A, B # here, A, B indicate arrays
    # selected to partition part, with padding for finite difference ops.

    # now we do whatever we want with them! We can edit them in-place, too

    r # return values are gathered in an array and returned.
end
```

In these arrays, the first index is always expected to correspond to the cell 
index.

Kwargs are passed as they are to the evaluation function.

Conversion functions `conv_to_backend` and `conv_from_backend` may be passed to
convert arrays (and partitions) to a custom array backend before any operations.

Example:

```julia
# for CuArrays:
using CUDA

conv_to_backend = x -> cu(x)
conv_from_backend = x -> Array(x)
```

`nthreads` may be specified to allow for multi-threading between partitions.

# Grid operators

```julia
u = rand(length(domain))
ux = similar(u)

domain(u, ux) do part, u, ux # values at local domain
    dx, dy = part.spacing |> eachcol
    # we have the very similar part.centers too ;)

    ux .= ( # note that we're editing in-place
        part(u, 1, 0) .- part(u, -1, 0)
    ) ./ (2 .* dx)
end

# ux is now the first, x-axis derivative of u
```

```julia
getalong(
    part::Partition, 
    U::AbstractArray, 
    dim::Int, i::Int
)
```

Obtain stencil index `i` along dimension `dim` in a partition array.
`part(u, 0, 3, 0)` is equivalent to `ibm.getalong(part, u, 2, 3)`, for example.

We also provide the following operators and utilities (check their docstrings!):

```julia
∇
Δ
δ
μ
MUSCL
laplacian_smoothing
stencil_average
advection
dissipation
divergent
```

# Boundary conditions

```julia
function impose_bc!(
    f,
    part::Partition, bname::String,
    args::AbstractArray...;
    impose_at_ghost::Bool = false,
    kwargs...
)
```

Impose boundary condition on domain array.

Example for non-penetration condition:

```julia
dom(u, v) do part, udom, vdom
    # function receives values of field properties at image points
    # and returns their values at the boundary
    ibm.impose_bc!(part, "wall", udom, vdom) do bdry, uimage, vimage
        nx, ny = bdry.normals |> eachcol
        un = @. nx * uimage + ny * vimage

        (
            uimage .- un .* nx,
            vimage .- un .* ny
        )
    end
end

# alternative return value:
uv = zeros(length(dom), 2)
uv[:, 1] .= 1.0
dom(uv) do part, uvdom
    ibm.impose_bc!(part, "wall", uvdom) do bdry, uvim
        uimage, vimage = eachcol(uvim)
        nx, ny = eachcol(bdry.normals)
        un = @. nx * uimage + ny * vimage

        uvim .- un .* bdry.normals
    end
end
```

Kwargs are passed directly to the BC function.
Note that other field variable args. may be passed
as auxiliary variables (e. g. the BC function may receive
3 arrays as an input, and return BCs solely for the first two).

We may directly return ghost cell values rather than boundary values
by activating flag `impose_at_ghost`.

```julia
struct Boundary{Ti, Tf}
    ghost_indices::AbstractVector{Ti}
    ghost_distances::AbstractVector{Tf}
    image_distances::AbstractVector{Tf}
    points::AbstractMatrix{Tf}
    normals::AbstractMatrix{Tf}
    image_interpolator::NNInterpolator.Accumulator
end
```

# Solving linear systems

You can solve linear systems using the `ImmersedBoundary.PointImplicit` module. Example for the Laplace equation:

```julia
using ImmersedBoundary.PointImplicit

mgrid = Multigrid(domain, 4) # 4 levels

function residual(u::AbstractVector)
    r = similar(u)

    domain(u, r) do part, u, r
        r .= (
            ∇(Δ(part, u, 1), 1) .+ ∇(Δ(part, u, 2), 2)
        ) # plus BC impositions, however you wish to do them
    end

    r
end

u = zeros(length(domain))
A, b, preconditioner = linearize(residual, u;
    n_hutchinson_samples = 20) # uses Hutchinson's trick to estimate diagonals

s, residual_ratio = solve(
    A, b, preconditioner;
    n_iter = 100, rtol = 1e-2, atol = 1e-7,
    multigrid = mgrid, verbose = true
)
u .+= s

# this works with more variables in the columns of a matrix as well.
```

# Postprocessing with surfaces

Surfaces may be used for postprocessing and coefficient integration. Example:

```julia
surf = dom.surfaces["wall"]

Cp_wall = surf(Cp) # interpolate array of field properties to wall

CX, CY = surface_integral(
    surf, Cp_wall .* surf.normals # surf.points is also available
)
```

Values are obtained an offset away from the surface (see `surf.offsets`) in order to obtain values like `τ` at the wall in wall-modelled simulations.

```julia
surf = dom.surfaces["wall"]

τ = μ .* surf(V) ./ surf.offsets # wall-normal gradient
```

To export, the kwarg `surface_data` is available in `export_vtk`:

```julia
export_vtk("destination", domain;
    surface_data = Dict(
        "wall" => (
            Cp = wall(Cp), # example
            τ = μ .* (
                wall(V)
            ) ./ surf.offsets
        ),
        "other_wall" => (
            #...
        )
    )
)
```

# CFD utilities

Check out the docstrings for the following functions and structs:

```julia
using ImmersedBoundary.CFD

Fluid
speed_of_sound
dynamic_viscosity
heat_conductivity
primitive2state
state2primitive
FlowBC
ISA_atmosphere
streamwise_direction
pressure_coefficient
inviscid_fluxes
viscous_fluxes
Reynolds_number
adjust_Reynolds
TimeAverage

using ImmersedBoundary.Turbulence

WallFunction
shear_rate
Smagorinsky_νSGS
Wray_Argawal

using ImmersedBoundary.IBL

m_closure
θ_closure
```