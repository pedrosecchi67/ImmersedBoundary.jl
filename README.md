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
struct Mesh
    origins::Matrix{Float64} # (ndims, ncells)
    widths::Matrix{Float64}
    centers::Matrix{Float64}
    in_domain::Vector{Bool}
    family_distances::Dict{String, Vector{Float64}}
    family_projections::Dict{String, Matrix{Float64}}
    families::Dict{String, Stereolitography}
end

function Mesh(
    origin::Vector{Float64}, widths::Vector{Float64},
    surfaces::Tuple{String, Stereolitography, Float64}...;
    interior_reference::Union{Vector{Float64}, Nothing} = nothing,
    boundary_surface::Union{Stereolitography, Nothing} = nothing,
    growth_ratio::Float64 = 1.1,
    ghost_layer_ratio::Float64 = 1.5,
    refinement_regions = [],
    hypercube_families = [],
    merge_tolerance::Real = 1e-7,
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
* A cell growth ratio;
* An interior point reference; and
* A ghost layer ratio, which defines the thickness of the ghost cell layer
    within a solid as a ratio of the local cell circumdiameter.

Family naming may be implemented by giving the same name string to several surfaces.
Surfaces without family names ("", empty strings) will be considered merely
a meshing resource, not a boundary.

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

# Domain definition 

```julia
function Domain(
    msh::Mesh;
    stencil = nothing,
    max_partition_size::Int = 1000_000,
    ghost_layer_ratio::Real = 1.5,
)
```

Constructor for a domain.

`ghost_layer_ratio` is the ratio between image point distances to the wall and the
ghost cells' circumradiameters.

Example:

```julia
dom = Domain(msh)

export_vtk("domain", dom) # check it out :)

@show ndims(dom) # dimensionality
@show length(dom) # number of cells
```

# Calculating residuals

```julia
(dom::Domain)(
    f, args::AbstractArray...; 
    n_threads::Int = 1,
    kwargs...
)
```

Run a loop over the partitions of a domain and
execute operations.

Example:

```julia
domain(r, u) do partition, rdom, udom
    # udom includes the parts of vector `u`
    # which affect the residual at partition `partition`.

    # now do some Cartesian grid operations and
    # update rdom
end

# after the loop, the values of `rdom` are returned to
# array `r`
```

This allows for large operations on field data
to be performed one partition at a time,
saving on max. memory usage.

Return values are also stored in a vector, which is
then returned.
Kwargs are passed to the called function.

Ran with `n_threads` treads.

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

The remaining fields of `Partition` include:

```julia
struct Partition
    index::Int64
    image::AbstractVector{Int64}
    image_in_domain::AbstractVector{Int64}
    skirt_indices::AbstractVector{Int64}
    domain::AbstractVector{Int64}
    stencils::AbstractDict
    centers::AbstractMatrix{Float64} # (ncells, ndims)
    spacing::AbstractMatrix{Float64} # (ncells, ndims)
    boundaries::Dict{String, Boundary}
end
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
```

# Boundary conditions

```julia
function impose_bc!(
    f,
    part::Partition, bname::String,
    args::AbstractArray{Float64}...;
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
struct Boundary
    points::AbstractMatrix{Float64} # (npoints, ndims)
    normals::AbstractMatrix{Float64} # (npoints, ndims)
    ghost_indices::AbstractVector{Int64}
    ghost_distances::AbstractVector{Float64}
    image_distances::AbstractVector{Float64}
    image_interpolator::NNInterpolator.Accumulator
end
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
```

# Custom array backends/GPUs

```julia
dom(part, Q) do part, Q
    # port partition to array backend
    cpart = to_backend(part, x -> CuArray(x))
    cQ = cu(Q)

    # run ops. on GPU

    Q .= Array(cQ)
end

# or, if your GPU supports the whole domain:
cdom = to_backend(dom, x -> CuArray(x))
```
