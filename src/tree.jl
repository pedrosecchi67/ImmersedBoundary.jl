using DocStringExtensions

using LinearAlgebra

using IterTools

"""
$TYPEDFIELDS

Struct to define a cell
"""
mutable struct TreeCell{T <: AbstractFloat, N}
    origin::Vector{T}
    widths::Vector{T}
    center::Vector{T}
    children::Array{TreeCell, N}
    index::Int64
    interior::Bool
    split_size::Int64
end

"""
$TYPEDSIGNATURES

Constructor for a cell from origin and widths
"""
function TreeCell(
    origin::AbstractVector,
    widths::AbstractVector;
    interior::Bool = true,
    split_size::Int64 = 2,
)

    nd = length(origin)
    
    children = Array{TreeCell, nd}(
        undef, fill(0, nd)...
    )

    center = @. origin + widths /  2

    origin = copy(origin)
    widths = copy(widths)

    TreeCell(
        origin, widths, center, children, 0, interior, split_size
    )

end

"""
$TYPEDSIGNATURES

Get number of cell dimensions
"""
Base.ndims(c::TreeCell) = length(c.origin)

"""
$TYPEDSIGNATURES

Split cell. Returns array of children
"""
function split!(c::TreeCell)

    c.children = Array{TreeCell}(undef, fill(c.split_size, ndims(c))...)

    w = c.widths ./ c.split_size

    for inds in Iterators.product(
        fill(1:c.split_size, ndims(c))...
    )
        mults = @. (inds - 1) / c.split_size

        o = c.origin .+ c.widths .* mults

        c.children[inds...] = TreeCell(o, w; interior = c.interior, split_size = c.split_size)
    end

    c.children

end

"""
$TYPEDSIGNATURES

Find if a cell is a leaf
"""
isleaf(c::TreeCell) = (
    length(c.children) == 0
)

"""
$TYPEDSIGNATURES

Collect all the leaves in a tree.

Filters to interior points if `interior_only` is true.
"""
function leaves(root::TreeCell; interior_only::Bool = false)

    if isleaf(root)
        if interior_only
            if root.interior
                return [root]
            else
                return TreeCell[]
            end
        else
            if root.index > 0 # case for cells clipped out of mesh by index
                return [root]
            else
                return TreeCell[]
            end
        end
    end

    if (root.index == -2) || (interior_only && (!root.interior))
        return TreeCell[]
    end

    mapreduce(
        l -> leaves(l; interior_only = interior_only), vcat, root.children
    )

end

"""
$TYPEDSIGNATURES

Find if a pair of cells has a shared face
"""
function is_neighbor(
    c1::TreeCell, c2::TreeCell;
    include_self::Bool = false,
    include_diagonal::Bool = false,
)

        mindist = minimum_distance(c1.center, c1.widths, c2.center, c2.widths)

        isneigh = let m = min(minimum(c1.widths), minimum(c2.widths)) * 0.001
            mindist < m
        end
        if !isneigh
            return false
        end

        if include_diagonal
            if include_self
                return true
            else
                return !(c1.center ≈ c2.center)
            end
        end

        if all(
                abs.(c1.center .- c2.center) .> (c1.widths .+ c2.widths) .* 0.999 ./ 2
        )
            return false
        end

        true

end

"""
$TYPEDSIGNATURES

Get all cells that neighbor a given cell through a shared face by
recursively visiting octree nodes
"""
function neighbors(root::TreeCell, c::TreeCell; include_diagonal::Bool = false,)

    if isleaf(root)
        if is_neighbor(root, c; include_diagonal = include_diagonal,)
            return [root]
        else
            return TreeCell[]
        end
    end

    f = filter(
        ch -> is_neighbor(ch, c; include_self = true, include_diagonal = include_diagonal),
        root.children
    )

    if length(f) == 0
        return TreeCell[]
    end

    mapreduce(
        ch -> neighbors(ch, c; include_diagonal = include_diagonal),
        vcat,
        f
    )

end

include("stereolitography.jl")
using .STLHandler

"""
$TYPEDSIGNATURES

Function to convert stereolitography objects to trees.
Leaves distance fields and trees untouched.
Warns the user that distance fields are advised if 3D
STLTrees are being constructed.
"""
function to_distance_metric(obj::Union{Stereolitography, STLTree, DistanceField}, 
        context::String)

    if obj isa Stereolitography
        if size(obj.points, 1) == 3
            @warn "3D stereolitography object passed for distance calculations in context $context. DistanceField(STLTree(stl), origin, widths) is recommended to avoid excessive costs."
        end

        return STLTree(obj)
    elseif obj isa STLTree
        if size(obj.stl.points, 1) == 3
            @warn "3D stereolitography dist. tree passed for distance calculations in context $context. DistanceField(tree, origin, widths) is recommended to avoid excessive costs."
        end

        return obj
    end

    obj # distance field (no warnings)

end

"""
$TYPEDSIGNATURES

Define `c.index` for each cell (recursively).

Non-leaf cells are given index 0
"""
function set_numbering!(
    root::TreeCell, N::Int = 0
)

    if isleaf(root)
        N += 1

        root.index = N

        return N
    end

    root.index = 0

    for ch in root.children
        N = set_numbering!(ch, N)
    end

    N

end

"""
$TYPEDSIGNATURES

Find the minimum possible distance between a pair of points within two boxes
"""
function minimum_distance(
    center1::AbstractVector, widths1::AbstractVector,
    center2::AbstractVector, widths2::AbstractVector,
)

    mindist = norm(
        (
            @. max(
                0.0,
                abs(center1 - center2) - (widths1 + widths2) / 2
            )
        )
    )

end

"""
$TYPEDSIGNATURES

Find the minimum possible distance between a box and a point
"""
function minimum_distance(
    center::AbstractVector, widths::AbstractVector,
    point::AbstractVector
)

    mindist = norm(
        (
            @. max(
                0.0,
                abs(center - point) - widths / 2
            )
        )
    )

end

"""
$TYPEDSIGNATURES

Find the maximum possible distance between a pair of points within two boxes
"""
function maximum_distance(
    center1::AbstractVector, widths1::AbstractVector,
    center2::AbstractVector, widths2::AbstractVector,
)

    maxdist = norm(
        (
            @. (
                abs(center1 - center2) + (widths1 + widths2) / 2
            )
        )
    )

end

"""
$TYPEDSIGNATURES

Define `interior` according to mask vector for each leaf.

Non-leaf cells are given index 0, or -1 if none of their children have 
`mask = true`
"""
function set_mask!(
    root::TreeCell, mask::AbstractVector
)

    if isleaf(root)
        m = mask[root.index]
        root.interior = m

        return m
    end
    
    # non-leaf: check for any interior leaf in children
    any_children = false
    for ch in root.children
        any_children = set_mask!(ch, mask) || any_children
    end

    if !any_children
        root.index = -1
    end

    any_children

end

"""
$TYPEDSIGNATURES

Re-index cells by marking interior cells as (sequential) integers
and exterior cells as 0.

Changes the index of non-leaf cells with no valid children to -2.
"""
function re_index!(root::TreeCell, _n::Int = 0)

    if isleaf(root)
        if root.interior
            _n += 1

            root.index = _n
        else
            root.index = 0
        end
    else
        nlast = _n

        for ch in root.children
            _n = re_index!(ch, _n)
        end

        if _n == nlast
            root.index = -2
        else
            root.index = 0
        end
    end

    _n

end

function _projections!(
    dists::AbstractVector,
    projs::AbstractMatrix,
    root::TreeCell,
    tree::Union{STLTree, DistanceField};
    interior_only::Bool = false,
    approximation_ratio::Real = 0.0
)

    if isleaf(root)
        # mask == false
        if interior_only && (!root.interior)
            return
        end

        # index specifies that the cell should not be included
        if root.index == 0
            return
        end

        p, d = tree(root.center)
        
        projs[:, root.index] .= p
        dists[root.index] = d
    else
        # no mask == true children
        if (root.index == -2) || (interior_only && root.index < 0)
            return
        end

        # beyond approximation ratio * circumradius?
        # truncate to same projection
        approx_projection = nothing
        if approximation_ratio > 0.0
            R = norm(root.widths) / 2
            p, d = tree(root.center)

            if d > R * approximation_ratio
                approx_projection = p
            end
        end

        if !isnothing(approx_projection)
            for lv in leaves(root)
                projs[:, lv.index] .= approx_projection
                dists[lv.index] = norm(approx_projection .- lv.center)
            end
        else # otherwise calculate
            for ch in root.children
                _projections!(
                    dists, projs,
                    ch, tree; interior_only = interior_only,
                    approximation_ratio = approximation_ratio
                )
            end
        end
    end

end

"""
$TYPEDSIGNATURES

Get number of leaves in a tree
"""
function ncells(
    root::TreeCell; interior_only::Bool = false
)

    if isleaf(root)
        return (
            interior_only ?
            root.interior : (root.index > 0) # even if interior_only is not specified, disregard cells with index 0 (cut out of mesh)
        )
    end

    # no valid leaves
    if (root.index == -2) || (interior_only && root.index == -1)
        return 0
    end

    sum(
        ch -> ncells(ch; interior_only = interior_only), root.children
    )

end

"""
$TYPEDSIGNATURES

Find the projections and distances of each cell's center on a stereolitography object.

The projection to the surface is approximated for cells that are far from the boundary
by distances greater than `norm(leaf.widths) * approximation_ratio`.
"""
function projections_and_distances(
    root::TreeCell,
    stl::Union{Stereolitography, STLTree, DistanceField};
    interior_only::Bool = false,
    approximation_ratio::Real = 0.0,
)

    tree = to_distance_metric(stl, "projections_and_distances")

    lvs = leaves(root)
    nc = length(lvs)

    dists = Vector{Float64}(undef, nc)
    projs = Matrix{Float64}(undef, ndims(root), nc)

    _projections!(
        dists, projs, root, tree; 
        interior_only = interior_only,
        approximation_ratio = approximation_ratio,
    )

    if interior_only
        isval = map(l -> l.interior, lvs)

        projs = projs[:, isval]
        dists = dists[isval]
    end

    (projs, dists)

end

"""
$TYPEDSIGNATURES

Get a vector of leaves that intersect with a given stereolitography object.

Find leaves in an octree that are in the vicinity of a stereolitography object,
with a distance to it no larger than `ratio * norm(widths(cell)) / 2`.

The default value of ratio is `sqrt(ndims(root))`
"""
function _intersections(
    root::TreeCell,
    tree::Union{STLTree, DistanceField};
    ratio = nothing,
    interior_only::Bool = false,
)

    if isnothing(ratio)
        ratio = sqrt(ndims(root))
    end

    if isleaf(root)
        # mask == false
        if interior_only && (!root.interior)
            return TreeCell[]
        end

        # torn off of mesh by index
        if root.index == 0
            return TreeCell[]
        end

        _, d = tree(root.center)

        threshold = ratio * norm(root.widths) / 2

        if d <= threshold
            return [root]
        else
            return TreeCell[]
        end
    end

    # no mask == true children
    if (root.index == -2) || (interior_only && root.index < 0)
        return TreeCell[]
    end

    threshold = ratio * norm(root.widths) / 2

    _, dist = tree(root.center)

    if dist > threshold
        return TreeCell[]
    end

    mapreduce(
        ch -> _intersections(ch, tree; ratio = ratio, interior_only = interior_only),
        vcat,
        root.children
    )

end

"""
$TYPEDSIGNATURES

Get a vector of leaves that intersect with a given stereolitography object.

Find leaves in an octree that are in the vicinity of a stereolitography object,
with a distance to it no larger than `ratio * norm(widths(cell)) / 2`.

The default value of ratio is `sqrt(ndims(root))`
"""
intersections(
    root::TreeCell,
    stl::Union{Stereolitography, STLTree, DistanceField};
    ratio = nothing,
    interior_only::Bool = false,
) = let tree = to_distance_metric(stl, "intersections")
    _intersections(root, tree; ratio = ratio, interior_only = interior_only)
end

"""
$TYPEDSIGNATURES

Recursively split the leaves of a tree (run N partitions)
"""
function recursive_split!(
    root::TreeCell, depth::Int = 1
)

    if isleaf(root)
        if depth == 0
            return
        end

        split!(root)
        for ch in root.children
            recursive_split!(ch, depth - 1)
        end
    else
        for ch in root.children
            recursive_split!(ch, depth)
        end
    end

end

"""
$TYPEDSIGNATURES

Balance an octree
"""
function balance!(
    root::TreeCell;
    include_diagonal::Bool = true,
)

    nref = 1
    while nref > 0
        set_numbering!(root)

        lvs = leaves(root)
        should_split = falses(length(lvs))
        for (il, lv) in enumerate(lvs)
            neighs = neighbors(root, lv; include_diagonal = include_diagonal,)

            ws = lv.widths

            for n in neighs
                nws = n.widths
                ni = n.index

                should_split[il] = should_split[il] || any(
                    ws .> nws .* (n.split_size + 1.0e-3)
                )
                should_split[ni] = should_split[ni] || any(
                    ws .< nws ./ (n.split_size + 1.0e-3)
                )
            end
        end

        nref = sum(should_split)

        for (lv, ss) in zip(lvs, should_split)
            if ss
                split!(lv)
            end
        end
    end

    set_numbering!(root)

end

"""
```
    struct Box
        origin::AbstractVector
        widths::AbstractVector
    end
```

Struct defining a refinement box
"""
struct Box
    origin::AbstractVector
    widths::AbstractVector
end

"""
```
    (b::Box)(pt::AbstractVector)
```

Distance to a box
"""
(b::Box)(pt::AbstractVector) = norm(
    (
        @. min(
            abs(pt - b.origin),
            abs(pt - b.origin - b.widths)
        ) * (pt - b.origin > b.widths || pt < b.origin)
    )
)

"""
```
    struct Ball
        center::AbstractVector
        radius::Real
    end
```

Struct to define a ball
"""
struct Ball
    center::AbstractVector
    radius::Real
end

"""
```
    (b::Ball)(pt::AbstractVector) = max(
        0.0,
        norm(b.center .- pt) - b.R
    )
```

Distance to a ball
"""
(b::Ball)(pt::AbstractVector) = max(
    0.0,
    norm(b.center .- pt) - b.radius
)

"""
```
    struct Line
        p1::AbstractVector
        p2::AbstractVector
        m::AbstractVector

        Line(p1::AbstractVector, p2::AbstractVector) = new(
            p1, p2,
            p2 .- p1
        )
    end
```

Struct to define a line
"""
struct Line
    p1::AbstractVector
    p2::AbstractVector
    m::AbstractVector

    Line(p1::AbstractVector, p2::AbstractVector) = new(
        p1, p2,
        p2 .- p1
    )
end

"""
```
    (l::Line)(pt::AbstractVector)
```

Distance to a line
"""
(l::Line)(pt::AbstractVector) = let ξ = l.m \ (pt .- l.p1)
    if ξ < 0.0
        return norm(pt .- l.p1)
    elseif ξ > 1.0
        return norm(pt .- l.p2)
    end

    norm(
        pt .- (l.p1 .+ l.m .* ξ)
    )
end

"""
$TYPEDSIGNATURES

Refine octree to have a maximum cell size at the boundary specified by the stereolitography surface
"""
function boundary_refine!(
    root::TreeCell,
    stl::Union{Stereolitography, STLTree, DistanceField},
    max_size::Real;
    ratio::Real = 2.0,
    buffer_layer_depth::Int = 2,
    recursive_split::Bool = false,
)

    tree = to_distance_metric(stl, "boundary_refine!")

    nref = 1

    max_size *= root.split_size ^ buffer_layer_depth

    while nref > 0
        set_numbering!(root)

        lvs = _intersections(
            root, tree;
            ratio = ratio,
        )

        nref = 0

        for l in lvs
            ms = maximum(l.widths)

            if ms > max_size
                split!(l)
                nref += 1
            end
        end
    end

    balance!(root; include_diagonal = true,)

    if recursive_split && buffer_layer_depth > 0
        recursive_split!(root, buffer_layer_depth)
    end

    set_numbering!(root)

end

"""
$TYPEDSIGNATURES

Refine octree to have a maximum cell size where the mesh intersects with a refinement region
given by a distance function
"""
function refine_region!(
    root::TreeCell,
    distance,
    max_size::Real;
    buffer_layer_depth::Int = 2,
    recursive_split::Bool = false,
    _recr::Bool = false,
)

    max_size *= root.split_size ^ buffer_layer_depth

    if maximum(root.widths) > max(
        max_size,
        distance(root.center)
    )
        if isleaf(root)
            split!(root)
        end

        for ch in root.children
            refine_region!(
                ch, distance, max_size;
                buffer_layer_depth = 0,
                recursive_split = false,
                _recr = true,
            )
        end
    end

    if !_recr
        balance!(root; include_diagonal = true,)

        if recursive_split && buffer_layer_depth > 0
            recursive_split!(root, buffer_layer_depth)
        end
    end

    set_numbering!(root)

end

"""
$TYPEDSIGNATURES

Refine an octree based on pairs or tuples of stereolitography objects
and their respective local length scales.

Refinement regions given by distance functions and max. length scales
are also recieved as a vector of pairs or tuples
"""
function refine!(
    root::TreeCell,
    boundaries...;
    refinement_regions::AbstractVector = [],
    buffer_layer_depth::Int = 3,
    ratio::Real = 2.0,
)

    for (stl, h) in boundaries
        boundary_refine!(
            root, stl, h; 
            buffer_layer_depth = buffer_layer_depth, recursive_split = false,
            ratio = ratio,
        )
    end

    for (dist, h) in refinement_regions
        refine_region!(
            root, dist, h;
            buffer_layer_depth = buffer_layer_depth, recursive_split = false,
        )
    end

    recursive_split!(root, buffer_layer_depth)

    set_numbering!(root)

    root

end

function _find_k_closest_cells!(
    cells::AbstractVector,
    dists::AbstractVector,
    root::TreeCell,
    x::AbstractVector,
    interior_only::Bool
)

    if isleaf(root)
        # invalid cell according to mask
        if interior_only && (!root.interior)
            return
        end

        # invalid cell according to index
        if root.index == 0
            return
        end

        d = norm(root.center .- x)

        i = length(cells) + 1
        while i > 1 && d < dists[i - 1]
            i -= 1
        end

        swap_d = d
        swap_c = root
        for k = i:length(cells)
            temp_c = cells[k]
            cells[k] = swap_c
            swap_c = temp_c

            temp_d = dists[k]
            dists[k] = swap_d
            swap_d = temp_d
        end
    else
        # no valid subcell
        if (root.index == -2) || (interior_only && root.index == -1)
            return
        end

        d = minimum_distance(root.center, root.widths, x)
        
        if d < dists[end]
            for ch in sort(
                vec(root.children);
                by = c -> minimum_distance(c.center, c.widths, x)
            )
                _find_k_closest_cells!(cells, dists, ch, x, interior_only)
            end
        end
    end

end

"""
$TYPEDSIGNATURES

Find `k` cells with centers closest to point `x`.

Gets `length(x) + 1` points if `k` is unspecified.

Return vector of cells and vector of distances.

`interior_only` may be specified to filter interior points.
"""
function find_k_closest_cells(
    root::TreeCell,
    x::AbstractVector,
    k::Int = 0;
    interior_only::Bool = false,
)

    if k == 0
        k = length(x) + 1
    end

    dists = fill(Inf, k)

    cells = Vector{Union{TreeCell, Nothing}}(undef, k)
    cells .= nothing

    _find_k_closest_cells!(cells, dists, root, x, interior_only)

    if any(isnothing, cells)
        isval = map(c -> !isnothing(c), cells)

        cells = cells[isval]
        dists = dists[isval]
    end

    (cells, dists)

end

"""
$TYPEDFIELDS

Struct to define an interpolation stencil
"""
struct Interpolator
    stencils::Matrix{Int64}
    weights::AbstractMatrix
end

"""
$TYPEDSIGNATURES

Constructor for an interpolation stencil.

Takes root cell `root` and a matix with each column
identifying an interpolation point.
"""
function Interpolator(
    root::TreeCell,
    X::AbstractMatrix;
    k::Int = 0,
    interior_only::Bool = false,
)

    k = max(size(X, 1) + 1, k)

    stencils = Matrix{Int64}(undef, k, size(X, 2))
    weights = Matrix{Float64}(undef, k, size(X, 2))

    for (j, x) in enumerate(eachcol(X))
        cls, dists = find_k_closest_cells(root, x, k; interior_only = interior_only,)

        stencils[:, j] .= map(
            c -> c.index,
            cls
        )
        weights[:, j] .= let ϵ = eps(eltype(dists))
            w = @. 1.0 / (dists + ϵ)

            w ./ sum(w)
        end
    end

    Interpolator(stencils, weights)

end

"""
$TYPEDSIGNATURES

Constructor for a linear interpolation stencil.

Takes root cell `root` and a matix with each column
identifying an interpolation point.
"""
function LinearInterpolator(
    root::TreeCell,
    X::AbstractMatrix;
    k::Int = 0,
    interior_only::Bool = false,
)

    k = max(size(X, 1) + 1, k)

    stencils = Matrix{Int64}(undef, k, size(X, 2))
    weights = Matrix{Float64}(undef, k, size(X, 2))

    for (j, x) in enumerate(eachcol(X))
        cls, dists = find_k_closest_cells(root, x, k; interior_only = interior_only)

        stencils[:, j] .= map(
            c -> c.index,
            cls
        )
        w = let ϵ = eps(eltype(dists))
            w = @. 1.0 / (dists + ϵ)
        end

        X = mapreduce(
            c -> [1.0 (c.center .- x)'],
            vcat,
            cls
        ) .* w

        weights[:, j] .= pinv(X)[1, :] .* w
    end

    Interpolator(stencils, weights)

end

"""
$TYPEDSIGNATURES

Run interpolator given a vector of cell center values
"""
function (intp::Interpolator)(u::AbstractVector)

    sts = intp.stencils
    ws = intp.weights

    dropdims(
        sum(
            u[sts] .* ws;
            dims = 1,
        );
        dims = 1
    )

end

"""
$TYPEDSIGNATURES

Run interpolation given an arbitrary array,
the last dimension of which corresponds to cell data
"""
(intp::Interpolator)(u::AbstractArray) = mapslices(
    intp,
    u;
    dims = ndims(u)
)

"""
$TYPEDSIGNATURES

Build an interpolation to stereolitography points
"""
Interpolator(
    root::TreeCell, stl::Stereolitography; k::Int = 0, interior_only::Bool = false,
) = Interpolator(
    root, stl.points; k = k, interior_only = interior_only,
)

"""
$TYPEDSIGNATURES

Build a linear interpolation to stereolitography points
"""
LinearInterpolator(
    root::TreeCell, stl::Stereolitography; k::Int = 0, interior_only::Bool = false,
) = LinearInterpolator(
    root, stl.points; k = k, interior_only = interior_only,
)

"""
$TYPEDSIGNATURES

Obtain graph with the connectivity structure of a given octree.

The graph is represented by a list of lists of neighbor indices
"""
function octree2graph(
    root::TreeCell; 
    include_diagonal::Bool = false,
    interior_only::Bool = false,
)

    lvs = leaves(root)
    mask = map(
        l -> ((!interior_only) || l.interior), lvs
    )

    graph = [
        [
            n.index for n in neighbors(root, c; include_diagonal = include_diagonal,)
        ] for c in lvs
    ]

    for neighs in graph
        filter!(
            n -> (n > 0 && mask[n]), neighs
        )
    end

    graph = graph[mask]

    graph

end

"""
Recursive private function to flag interior points without going down
branches that are completely within/without the domain
"""
function _flag_interior!(
    interior,
    root::TreeCell,
    stltree::Union{STLTree, DistanceField},
    interior_reference::AbstractVector{Float64},
    ratio::Real,
)

    D = norm(root.widths) # circumdiameter
    nd = ndims(root)

    _, d = stltree(root.center)
    sd = let sgn = (
        1 - 2 * point_in_polygon(
            stltree, root.center;
            outside_reference = interior_reference
        )
    )
        sgn * d
    end
    threshold = D * sqrt(nd) * ratio

    if isleaf(root)
        interior[root.index] = interior[root.index] && (sd >= threshold)
    else
        # might have a leaf inside the domain?
        only_inside = sd - (threshold > 0.0 ? threshold : 0.0) - D / 2 * 1.1 > 0.0
        only_outside = - sd - (threshold < 0.0 ? - threshold : 0.0) - D / 2 * 1.1 > 0.0

        if only_inside
            return
        elseif only_outside
            for lv in leaves(root)
                interior[lv.index] = false
            end
        else # continue recursively
            for ch in root.children
                _flag_interior!(
                    interior, ch,
                    stltree, interior_reference, ratio
                )
            end
        end
    end

end

"""
$TYPEDSIGNATURES

Find mask to points in the interior of a domain
separated from the outside by a set of stereolitographies.

If interior point vector `interior` is not defined, then the root cell
origin is assumed to belong to the interior of the domain.

If `re_index` is true (default), the cells in the mesh will
be re-indexed so that exterior cells are given index zero and can no longer
be accessed.

`ratio` is such that the cell is flagged as interior if signed distance function `d > ratio * norm(widths) * sqrt(ndims(msh))`.
"""
function clip_interior!(
    root::TreeCell,
    bdries::Union{Stereolitography, STLTree, DistanceField}...;
    interior = nothing,
    re_index::Bool = true,
    ratio::Real = 0.0,
)

    ncls = set_numbering!(root)

    if isnothing(interior)
        interior = root.origin .- 0.1 .* root.widths
    end

    is_interior = trues(ncls)
    for stl in bdries
        tree = to_distance_metric(stl, "clip_interior!")

        _flag_interior!(
            is_interior,
            root,
            tree,
            interior,
            ratio
        )
    end

    set_mask!(root, is_interior)
    if re_index
        re_index!(root)
    end

    is_interior

end

using WriteVTK

"""
$TYPEDSIGNATURES

Write octree to VTK file using WriteVTK.
Saves vectors passed as kwargs to the grid file.
"""
function octree2vtk(
    fname::String,
    root::TreeCell;
    interior_only::Bool = false,
    kwargs...
)

    lvs = leaves(root)

    mask = map(
        c -> ((!interior_only) || c.interior) && (c.index > 0), lvs
    )

    nd = ndims(root)
    ncorners = 2 ^ nd

    ctype = (
        nd == 2 ? VTKCellTypes.VTK_PIXEL : VTKCellTypes.VTK_VOXEL
    )

    multipliers = mapreduce(
        collect, hcat,
        Iterators.product(
            fill((0, 1), nd)...
        )
    )

    points = mapreduce(
        c -> multipliers .* c.widths .+ c.origin,
        hcat,
        lvs
    )

    k = 0
    cells = MeshCell[]
    _conn = collect(1:ncorners)
    for l in lvs
        conn = _conn .+ (k * ncorners)

        k += 1

        push!(
            cells,
            MeshCell(ctype, conn)
        )
    end

    cells = cells[mask]

    grid = vtk_grid(fname, points, cells)

    if isnothing(mask)
        for (k, v) in kwargs
            grid[String(k)] = Float64.(v)
        end
    else
        for (k, v) in kwargs
            grid[String(k)] = Float64.(
                (
                    length(mask) == size(v, ndims(v)) ?
                    selectdim(
                        v, ndims(v), mask
                    ) :
                    v
                )
            )
        end
    end

    # vtk_save(grid)
    grid

end

"""
$TYPEDSIGNATURES

Get cells in the boundary of a hypercube
"""
function boundary_cells(
    root::TreeCell, dim::Int, front::Bool; interior_only::Bool = false,
)

    if isleaf(root)
        # invalid cell
        if ((!interior_only) || root.interior) && (root.index > 0)
            return [root]
        else
            return TreeCell[]
        end
    end

    # no valid subcells
    if (root.index == -2) || (interior_only && root.index == -1)
        return TreeCell[]
    end

    mapreduce(
        ch -> boundary_cells(ch, dim, front; interior_only = interior_only),
        vcat,
        selectdim(root.children, dim, (front ? root.split_size : 1))
    )

end

"""
$TYPEDSIGNATURES

Get projections and distances to a hypercube boundary
"""
function boundary_proj_and_dist(
    root::TreeCell, dim::Int, front::Bool;
    interior_only::Bool = false,
)

    lvs = leaves(root; interior_only = interior_only)

    X = mapreduce(
        c -> c.center, hcat, lvs
    )

    projs = copy(X)
    projs[dim, :] .= (
        root.origin[dim] + root.widths[dim] * front
    )

    dists = map(
        norm, eachcol(projs .- X)
    )

    (projs, dists)

end

"""
$TYPEDSIGNATURES

Get the depth of a tree
"""
function depth(root::TreeCell; interior_only::Bool = false,)

    if isleaf(root)
        return 0
    end

    # no valid subcells
    if (root.index == -2) || (interior_only && root.index == -1)
        return 0
    end

    maximum(
        ch -> depth(ch; interior_only = interior_only), root.children
    ) + 1

end

"""
$TYPEDSIGNATURES

Find maximum number of cells, including non-leaf nodes 
"""
max_cells(root::TreeCell; interior_only::Bool = false) = (
    isleaf(root) ?
    (((!interior_only) || root.interior) && root.index > 0) :
    (
        (root.index == -2) || (interior_only && root.index == -1) ?
        0 :
        sum(
            ch -> max_cells(ch; interior_only = interior_only),
            root.children
        ) + 1
    )
)

"""
$TYPEDSIGNATURES

Find largest possible blocks with leaves of a constant cell size.

Returns a vector of cells.

An optional max. depth may be specified.
"""
function blocks(
    root::TreeCell, max_depth::Int = 1000;
    _store! = nothing,
    interior_only::Bool = false,
)

    @assert max_depth > 0 "Pedro, you screwed up with the blocks function. Make max_depth > 0"

    recursive = !isnothing(_store!)

    if !recursive
        # edge case for root as leaf:
        if isleaf(root)
            return [root]
        end

        # store blocks in pre-allocated vector, then cut and return

        storage = Vector{TreeCell}(undef, max_cells(root))
        cnt = 0

        _store! = c -> begin
            cnt += 1
            storage[cnt] = c
        end

        blocks(root, max_depth; _store! = _store!, interior_only = interior_only)

        if interior_only
            return filter(
                c -> (isleaf(root) && root.interior) || c.index >= 0,
                storage[1:cnt]
            )
        else
            return storage[1:cnt]
        end
    end

    if isleaf(root)
        # invalid cell
        if ((!interior_only) || root.interior) && (root.index > 0)
            return (0, 0)
        else
            return (-1, -1) # terminate without registering
        end
    end

    # no valid subcells: terminate
    if (root.index == -2) || (interior_only && root.index == -1)
        return (-1, -1)
    end

    rngs = map(
        ch -> blocks(ch, max_depth; _store! = _store!, interior_only = interior_only),
        root.children
    )

    # find range
    dmin = minimum(r -> r[1], rngs)
    dmax = maximum(r -> r[2], rngs)

    # already terminated
    if dmax < 0
        return (-1, -1)
    end

    if (dmin != dmax) || dmin < 0 || dmax >= max_depth # termination
        # store block-like cells
        for ((dmi, dma), c) in zip(rngs, root.children)
            if dmi == dma && dmi >= 0
                _store!(c)
            end
        end

        # return termination signal
        return (-1, -1)
    end

    (dmin + 1, dmin + 1)

end

"""
$TYPEDSIGNATURES

Find leaf in which a point is located. Returns nothing if not found
"""
function find_leaf(
    root::TreeCell, x::AbstractVector;
    interior_only::Bool = false,
)

    # termination
    if isleaf(root)
        if (interior_only && (!root.interior)) || (root.index == 0)
            return nothing
        end
    else
        if (root.index == -2) || (interior_only && root.index == -1)
            return nothing
        end
    end

    ξ = (x .- root.origin) ./ root.widths

    if any(
        xi -> xi < 0.0 || xi >= 1.0,
        ξ
    )
        return nothing
    end

    if isleaf(root)
        return root
    end

    ret = nothing
    for ch in root.children
        r = find_leaf(ch, x; interior_only = interior_only)

        if !isnothing(r)
            ret = r
        end
    end

    ret

end

"""
Recursively split a tree until its leaves are
one ref. level coarser than the previous, reference tree.
"""
function _coarsen_split!(
    tref::TreeCell, tree::TreeCell
)

    if isleaf(tref)
        return
    end

    if any(
        isleaf, tref.children
    ) # cut refinement here!
        tree.interior = any(
               l -> l.interior, leaves(tref)
        )

        return
    end

    split!(tree)

    for (tch, ch) in zip(tref.children, tree.children)
        _coarsen_split!(tch, ch)
    end

end

"""
$TYPEDSIGNATURES

Obtain tree one level coarser than the reference.
"""
function coarsen(reference_tree::TreeCell)

    tree = TreeCell(
        reference_tree.origin, reference_tree.widths;
        split_size = reference_tree.split_size
    )

    _coarsen_split!(reference_tree, tree)

    # numbering and masking
    set_numbering!(tree)

    mask = map(
        l -> l.interior, leaves(tree)
    )
    set_mask!(tree, mask)
    re_index!(tree)

    tree

end

