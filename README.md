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
    function Mesh(
        origin::AbstractVector, widths::AbstractVector,
        surfaces::Tuple...;
        growth_ratio::Real = 2.0f0,
        tolerance::Real = 1f-7,
        block_size::Int = 8,
        refinement_regions::AbstractVector = [],
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

Example:

```julia
stl = Stereolitography("rae2822.dat")
features = feature_regions(stl; radius = 0.05)
feature_dfield = DistanceField(features)

msh = Mesh(
    [-40.0, -40.0], # hypercube origin
    [80.0, 80.0], # hypercube widths
    ("wall", stl, 4e-3);
    refinement_regions = [
        (feature_dfield, 0.5e-3),
    ],
    verbose = true
)

@show length(msh) # number of cells
```

# Creating domain

You can turn a mesh into a domain with all the information necessary for residual computation with:

```julia
dom = Domain(
    msh;
    max_partition_size = 100_000, # max. number of cells per partition
    partition_skirt_depth = 2, # partition "skirt" depth. Should be 2 for second order ops.
    ghost_layer_ratio = 1.5f0, # ratio between cell circumdiameter and max. ghost layer boundary distance
    hypercube_families = [
        "inlet" => [
            (1, false), # dimension, front/back
            (2, false), (2, true),
            (3, false), (3, true)
        ],
        "outlet" => [
            (1, true)
        ]
    ],
)

@show length(dom) # number of cells
@show ndims(dom) # number of dom. dimensions
```

# Calculating residuals

```julia
function (dom::Domain)(
    f,
    args::AbstractArray...; 
    conv_to_backend = identity,
    conv_from_backend = identity,
    n_threads::Int = 0, # default: as many as available
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

dom(args...;
    conv_to_backend = conv_to_backend,
    conv_from_backend = conv_from_backend) do args...
    # ...
end
```

# Grid operators

```julia
u = rand(length(domain))
ux = similar(u)

domain(r, u) do part, r, u # values at local domain
    dx, dy = part.spacing |> eachcol
    # we have the very similar part.centers too ;)

    # cell Green-Gauss gradient
    ux = cell_gradient(part, u, 1)

    # similar, but returns tuple with each dimension
    ∇u = cell_gradient(part, u)


    # example for linear advection-dissipation
    C = [1.0f0, 1.0f0]
    a = 1f-3

    r .= 0
    D = JST_sensor(part, u)
    for dim = 1:ndims(part)
        uL, uR = MUSCL(
            part, u, ∇u[dim], # cell value, grad. at cell center
            dim; 
            D = D, # optional shock sensor
            high_order = true # if true, switches to fourth order reconstruction
            # at smooth regions
        )

        # advection
        r .-= green_gauss(
            part, uL .* C[dim], # upwind
            dim
        )

        # unsigned_green_gauss also provides an integral without regard for flow directions:
        CFL .+= unsigned_green_gauss(
            part, fill(C[dim], length(uL)), dim
        )

        # dissipation
        r .+= green_gauss(
            part,
            a .* face_gradient(part, u, dim), dim
        )

        # alternative with face gradient orthogonal corrections
        ∇uf = face_gradient(part, u, ∇u, dim) # returns tuple with gradient along each dimension
        r .+= green_gauss(
            part,
            a .* ∇uf[dim], dim
        )
    end
end
```

Check out the docstrings for utility `divergent()` as well!

# Boundary conditions

```julia
function impose_bc!(
    f,
    dom::Domain, bname::String,
    args::AbstractArray...;
    kwargs...
)
```

Impose boundary condition on domain array.

Example for non-penetration condition:

```julia
# function receives values of field properties at image points
# and returns their values at the boundary
ibm.impose_bc!(dom, "wall", u, v) do bdry, uimage, vimage
    nx, ny = bdry.normals |> eachcol
    un = @. nx * uimage + ny * vimage

    (
        uimage .- un .* nx,
        vimage .- un .* ny
    )
end

# alternative return value:
uv = zeros(length(dom), 2)
uv[:, 1] .= 1.0
ibm.impose_bc!(dom, "wall", uv) do bdry, uvim
    uimage, vimage = eachcol(uvim)
    nx, ny = eachcol(bdry.normals)
    un = @. nx * uimage + ny * vimage

    uvim .- un .* bdry.normals
end
```

Kwargs are passed directly to the BC function.
Note that other field variable args. may be passed
as auxiliary variables (e. g. the BC function may receive
3 arrays as an input, and return BCs solely for the first two).

Data for BC calculation includes:

```julia
struct Boundary{Ti <: Integer, Tf <: AbstractFloat}
    ghost_indices::AbstractVector{Ti}
    projections::AbstractMatrix{Tf}
    normals::AbstractMatrix{Tf}
    image_distances::AbstractVector{Tf}
    ghost_distances::AbstractVector{Tf}
end
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

wall_function
shear_rate
Smagorinsky_νSGS
WALE_νSGS
Wray_Agarwal
standard_kϵ
Ducros_sensor

using ImmersedBoundary.IBL

m_closure
θ_closure

using ImmersedBoundary.Solver

FAS!
```

# Other types in custom array backends

Most data types support method `to_backend`:

```julia
using CUDA: cu

bc = FlowBC(fluid, P)
bc = ImmersedBoundary.to_backend(bc, x -> cu(x))
```
