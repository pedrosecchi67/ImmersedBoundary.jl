module STLHandler

    using DocStringExtensions

    using DelimitedFiles

    using WriteVTK

    using Statistics
    using LinearAlgebra

    export Stereolitography, STLTree, DistanceField, point_in_polygon, stl2vtk, refine_to_length
        
    module STLReader

        function is_ascii(file_path::String)
            # Open the file in read mode
            first_string = open(file_path, "r") do file
                # Read the first 5 characters from the file
                first_chars = read(file, 5)

                # Convert the read bytes to a string
                first_string = String(first_chars)
            end

            return first_string == "solid"
        end

        function read_stl_ascii(filename::String)
            vertices = Vector{Vector{Float64}}()
            faces = Vector{Vector{Int64}}()

            face = Int64[]
            open(filename, "r") do file
                for _line in eachline(file)
                    line = strip(_line)

                    if startswith(line, "vertex")
                        # Extract vertex coordinates
                        coords = split(line)
                        x = parse(Float64, coords[2])
                        y = parse(Float64, coords[3])
                        z = parse(Float64, coords[4])
                        push!(vertices, [x, y, z])
                        push!(face, length(vertices))
                    elseif startswith(line, "facet normal")
                        # Start of a new face
                        face = Vector{Int64}()
                    elseif startswith(line, "endloop")
                        # End of a face, add it to the faces list
                        push!(faces, face)
                    end
                end
            end

            vertices = reduce(hcat, vertices)
            faces = reduce(hcat, faces)

            return vertices, faces
        end

        function read_stl_binary(filename::String)
                contents = open(filename, "r") do file
                    read(file)
                end

                N0 = 0
                popN = N -> let r = (N0+1):(N0+N)
                    v = contents[r]
                    N0 += N

                    v
                end

                # header
                _ = popN(80)

                # number of tris
                ntri = reinterpret(UInt32, popN(4))[1] |> Int64

                points = zeros(Float64, 3, 3 * ntri)
                simplices = zeros(Int64, 3, ntri)

                for k = 1:ntri
                    _ = popN(12) # normal

                    points[:, 3*(k-1)+1] .= Float64.(reinterpret(Float32, popN(12)))
                    points[:, 3*(k-1)+2] .= Float64.(reinterpret(Float32, popN(12)))
                    points[:, 3*(k-1)+3] .= Float64.(reinterpret(Float32, popN(12)))

                    simplices[:, k] .= (3*(k-1)+1):(3*(k-1)+3)

                    _ = popN(2)
                end

                (points, simplices)
        end

        """
        ```
            read_stl(filename::String)
        ```

        Read STL file and return a matrix of points (shape (3, ...)) and a matrix of
        simplices (shape (3, ...)).
        """
        function read_stl(filename::String)

            if is_ascii(filename)
                return read_stl_ascii(filename)
            end

            read_stl_binary(filename)

        end

    end
    using .STLReader: read_stl

    """
    $TYPEDFIELDS

    Struct to represent stereolitography data

    Each column in `points` represents a point in space, and each 
    column in `simplices`, the point indicesfor a simplex face
    """
    struct Stereolitography
        points::AbstractMatrix{Float64}
        simplices::AbstractMatrix{Int64}
    end

    """
    $TYPEDSIGNATURES

    Constructor for Stereolitography that builds a (closed if `closed = true`)
    two-dimensional surface from a set of points (matrix `(ndims, npts)`)
    """
    function Stereolitography(
        points::AbstractMatrix;
        closed::Bool = false,
    )

        if closed
            if norm(points[:, 1] .- points[:, end]) < sqrt(eps(eltype(points)))
                closed = false
            end
        end

        simplices = let inds = collect(1:size(points, 2))
            [
                inds';
                circshift(inds, -1)'
            ]
        end

        if !closed
            simplices = simplices[:, 1:(end - 1)]
        end

        Stereolitography(points, simplices)

    end

    """
    $TYPEDSIGNATURES

    Obtain stereolitography data from mesh file.

    Uses MeshIO.jl as a backend.

    Disconsiders any mesh elements that aren't triangles.

    If a .dat file is provided, it will be interpreted as a Selig-format dat file
    describing a closed two-dimensional surface. It should include no header
    """
    function Stereolitography(
        fname::String,
    )

        if fname[(end - 3):end] in (".dat", ".DAT") # Selig format airfoil
            return Stereolitography(permutedims(readdlm(fname)); closed = true)
        end

        points, simplices = read_stl(
            fname
        )

        Stereolitography(points, simplices)

    end

    """
    $TYPEDSIGNATURES

    "Concatenate" stereolitography objects into a single
    struct
    """
    Base.cat(
        stl::Stereolitography...
    ) = Stereolitography(
        mapreduce(
            s -> s.points, hcat, stl
        ),
        mapreduce(
            s -> s.simplices, hcat, stl
        ),
    )

    """
    $TYPEDSIGNATURES

    Export STL surfaces to VTK format.
           
    ".vtu" extension will be added to `fname`
    """
    function stl2vtk(
        fname::String,
        surf::Stereolitography;
        kwargs...
    )   
       
        pts = surf.points

        ctype = (
            size(pts, 1) == 3 ?
            WriteVTK.VTKCellTypes.VTK_TRIANGLE :
            WriteVTK.VTKCellTypes.VTK_LINE
        )
         
        cells = map(
            simp -> WriteVTK.MeshCell(ctype, copy(simp)),
            eachcol(surf.simplices)
        )
                
        grid = vtk_grid(
            fname, pts, cells
        )
         
        for (vname, u) in kwargs
            grid[String(vname)] = u
        end

        # vtk_save(grid)
        grid
            
    end

    """
    $TYPEDFIELDS

    Struct to define a bounding box
    """
    struct BoundingBox
        origin::AbstractVector{Float64}
        widths::AbstractVector{Float64}
    end

    """
    $TYPEDSIGNATURES

    Obtain a bounding box from an STL object
    """
    function BoundingBox(stl::Stereolitography)

        mins = map(
            dims -> let simps = stl.simplices
                minimum(
                    ipt -> let inds = view(simps, ipt, :)
                        x = view(stl.points, dims, inds)

                        minimum(x)
                    end,
                    1:size(simps, 1)
                )
            end,
            1:size(stl.points, 1)
        )

        maxs = map(
            dims -> let simps = stl.simplices
                maximum(
                    ipt -> let inds = view(simps, ipt, :)
                        x = view(stl.points, dims, inds)

                        maximum(x)
                    end,
                    1:size(simps, 1)
                )
            end,
            1:size(stl.points, 1)
        )

        origin = mins
        widths = maxs .- mins

        BoundingBox(origin, widths)

    end

    """
    $TYPEDSIGNATURES

    Split stereolitography object at plane traced through the median
    of simplex centers.

    Also returns index of best split dimension, and the location of the
    split plane in said dimension.
    """
    function split_at_plane(stl::Stereolitography)

        centers = dropdims(
            sum(
                view(stl.points, :, stl.simplices);
                dims = 2
            ); dims = 2
        ) ./ size(stl.simplices, 1)

        coords = eachrow(centers)

        planes = map(median, coords)
        splits = map(
            (plane, u) -> (
                (@. u <= plane),
                (@. u > plane)
            ),
            planes, coords
        )

        metric = map(
            sep -> let (ls, rs) = sep
                min(
                    1.0 - sum(ls) / length(ls),
                    1.0 - sum(rs) / length(rs),
                )
            end,
            splits
        )
        dim = argmax(metric)
        left_simplices, right_simplices = findall.(splits[dim])
        plane = planes[dim]

        (
            Stereolitography(
                stl.points, view(stl.simplices, :, left_simplices)
            ),
            Stereolitography(
                stl.points, view(stl.simplices, :, right_simplices)
            ),
            dim, plane
        )

    end

    """
    $TYPEDSIGNATURES

    Find minimum distance to bounding box
    """
    function minimum_distance(box::BoundingBox, x::AbstractVector{Float64})

        proj = @. clamp(x, box.origin, box.widths + box.origin)

        norm(proj .- x)

    end

    """
    $TYPEDSIGNATURES

    Project point onto a simplex
    """
    function proj2simplex(
        simplex::AbstractMatrix, point::AbstractVector
    )

        if size(simplex, 2) == 1
            return vec(simplex)
        elseif size(simplex, 2) == 2 # optimized for lines
            p0 = simplex[:, 1]
            u = simplex[:, 2] .- p0

            ξ = clamp(
                ((point .- p0) ⋅ u) / (u ⋅ u + eps(eltype(simplex))),
                0.0, 1.0
            )

            return p0 .+ u .* ξ
        end

        p0 = simplex[:, 1]
        M = simplex[:, 2:end] .- p0

        dp = (point .- p0)
        x = M \ dp

        nd = size(simplex, 2)
        if sum(x) > 1.0 || any(x .< 0.0)
            closest_proj = similar(dp)
            dist = Inf

            for i = 1:nd
                face = simplex[:, setdiff(1:nd, i)]

                proj = proj2simplex(face, point)
                d = norm(proj .- point)

                if d < dist
                    closest_proj .= proj
                    dist = d
                end
            end

            return closest_proj
        end

        p0 .+ (M * x)

    end

    """
    $TYPEDSIGNATURES

    Obtain projection of a point onto a stereolitography object
    """
    function proj2stl(stl::Stereolitography, x::AbstractVector)
        minproj = similar(x)
        mindist = Inf

        for simp in eachcol(stl.simplices)
                proj = proj2simplex(stl.points[:, simp], x)
                d = norm(x .- proj)

                if d < mindist
                        mindist = d
                        minproj .= proj
                end
        end

        return minproj
    end

    """
    $TYPEDFIELDS

    Struct to define a stereolitography distance tree
    node.
    """
    struct STLTree
        left_child::Union{STLTree, Nothing}
        right_child::Union{STLTree, Nothing}
        stl::Stereolitography
        box::BoundingBox
    end

    """
    $TYPEDSIGNATURES

    Find if an STL tree node is a leaf.
    """
    isleaf(node::STLTree) = isnothing(node.left_child)

    """
    $TYPEDSIGNATURES

    Build an STL distance tree from a stereolitography object
    """
    function STLTree(stl::Stereolitography; leaf_size::Int64 = 1)

        box = BoundingBox(stl)

        if size(stl.simplices, 2) <= leaf_size
            return STLTree(
                nothing, nothing,
                stl, box
            )
        end

        lstl, rstl, _, _ = split_at_plane(stl)

        STLTree(
            STLTree(lstl; leaf_size = leaf_size), STLTree(rstl; leaf_size = leaf_size),
            stl, box
        )

    end

    """
    $TYPEDSIGNATURES

    Find projection and distance to an STL distance tree
    """
    function projection_and_distance(
        node::STLTree, x::AbstractVector{Float64},
        p::Union{Nothing, AbstractVector{Float64}} = nothing,
        d::Real = Inf64
    )

        if isnothing(p)
            p = fill(Inf64, length(x))
        end

        if isleaf(node)
            _p = proj2stl(node.stl, x)
            _d = norm(_p .- x)

            if _d < d
                d = _d
                p = _p
            end

            return (p, d)
        end

        mdist_left = minimum_distance(node.left_child.box, x)
        mdist_right = minimum_distance(node.right_child.box, x)

        if mdist_left > d && mdist_right > d
            return (p, d)
        end

        if mdist_left < mdist_right
            p, d = projection_and_distance(node.left_child, x, p, d)

            if mdist_right < d
                p, d = projection_and_distance(node.right_child, x, p, d)
            end
        else
            p, d = projection_and_distance(node.right_child, x, p, d)

            if mdist_left < d
                p, d = projection_and_distance(node.left_child, x, p, d)
            end
        end

        (p, d)

    end

    """
    Get simplex normal
    """
    function normal(simplex::AbstractMatrix{Float64})

        ϵ = eps(eltype(simplex))

        if size(simplex, 1) == 2
            v = simplex[:, 2] .- simplex[:, 1]

            return [
                - v[2], v[1]
            ] ./ (norm(v) + ϵ)
        end

        p0 = simplex[:, 1]

        n = cross(simplex[:, 2] .- p0, simplex[:, 3] .- p0)

        n ./ (norm(n) + ϵ)

    end

    """
    Find if a line connecting two points crosses a simplex
    """
    function crosses_simplex(
        simplex::AbstractMatrix{Float64},
        p1::AbstractVector{Float64},
        p2::AbstractVector{Float64},
    )

        p0 = simplex[:, 1]

        nϵ = sqrt(eps(eltype(p0)))

        dp = (p2 .- p1)

        n = normal(simplex)
        dp .+= let p = n ⋅ dp
            n .* (
                p < 0.0 ?
                - nϵ :
                nϵ
            )
        end

        M = [(simplex[:, 2:end] .- p0) dp]

        M = pinv(M)

        ξ1 = M * (p1 .- p0)
        ξ2 = M * (p2 .- p0)

        if ξ1[end] * ξ2[end] > - nϵ
            return false
        end

        ξ1 = ξ1[1:(end - 1)]

        if any(
            x -> x < - nϵ,
            ξ1
        )
            return false
        end

        if sum(ξ1) > 1.0 + nϵ
            return false
        end

        true

    end

    """
    $TYPEDSIGNATURES

    Obtain projection and distance to stereolitography distance
    tree.
    """
    (tree::STLTree)(x::AbstractVector{Float64}) = projection_and_distance(tree, x)

    """
    $TYPEDSIGNATURES

    Obtain number of times a line connecting two points crosses
    a simplex in a given stereolitography object.
    """
    n_crossings(
        stl::Stereolitography, 
        p1::AbstractVector{Float64},
        p2::AbstractVector{Float64},
    ) = map(
        simp -> crosses_simplex(view(stl.points, :, simp), p1, p2),
        eachcol(stl.simplices)
    ) |> sum

    """
    $TYPEDSIGNATURES

    Check whether a line segment intersects with a box
    """
    function intersects(
        box::BoundingBox,
        p1::AbstractVector{Float64}, p2::AbstractVector{Float64}
    )

        ϵ = eltype(p1) |> eps |> sqrt

        box_lb = copy(box.origin)
        box_ub = @. box.origin + box.widths

        @. box_ub -= box_lb
        p1 = p1 .- box_lb
        p2 = p2 .- box_lb
        @. box_lb -= box_lb # zero

        u = @. p2 - p1

        # coordinates along u for intersection with box wall planes:
        ξ = [
             (@. (box_lb - p1) * sign(u) / (abs(u) + ϵ));
             (@. (box_ub - p1) * sign(u) / (abs(u) + ϵ))
        ]
        clamp!(ξ, - ϵ, 1.0 + ϵ) # clamp to values on the line segment

        intersection_points = u .* ξ' .+ p1

        # check if any of these intersection points is actually on the box:
        any(
                all,
                eachcol(
                        @. (
                            intersection_points >= box_lb - box.widths * 0.001 && 
                            intersection_points <= box_ub + box.widths * 0.001
                        )
                )
        )

    end

    """
    $TYPEDSIGNATURES

    Find number of times a line connecting two points crosses
    the simplices in an STL distance tree.
    """
    function n_crossings(
        node::STLTree,
        p1::AbstractVector{Float64},
        p2::AbstractVector{Float64},
    )

        if isleaf(node)
            return n_crossings(node.stl, p1, p2)
        end

        n = 0

        if intersects(node.left_child.box, p1, p2)
            n += n_crossings(node.left_child, p1, p2)
        end

        if intersects(node.right_child.box, p1, p2)
            n += n_crossings(node.right_child, p1, p2)
        end

        n

    end

    """
    $TYPEDSIGNATURES

    Point-in-polygon query for an STL distance tree.

    Requires a watertight surface without zero-area elements.

    Takes an optional outsize point reference and works with ray tracing.
    """
    function point_in_polygon(
        node::STLTree, x::AbstractVector{Float64};
        outside_reference = nothing,
    )

        if isnothing(outside_reference)
            outside_reference = node.box.origin .- node.box.widths .* 0.1
        end

        n_crossings(node, outside_reference, x) % 2 == 1

    end

    """
    Run an iteration of refinement by cutting the simplices with an edge 
    larger than a given threshold.

    Returns the new stereolitography struct and 
    the number of split simplices.
    """
    function _cut_larger(
        stl::Stereolitography,
        Lmax::Float64
    )

        points = stl.points
        simplices = copy(stl.simplices)

        isfree = trues(size(points, 2))

        new_pts = []
        new_simps = []

        # get indices of points and add their average, returning new pt. index
        add_point! = (i, inext) -> let x = (points[:, i] .+ points[:, inext]) ./ 2
            push!(new_pts, x)
            
            size(points, 2) + length(new_pts)
        end
        # get view to simplex, i and inext, add a simplex with a new point between them
        split_simplex! = (simpview, i, inext) -> let newsimp = copy(simpview)
            isfree[simpview] .= false

            newpt = add_point!(simpview[i], simpview[inext])

            newsimp[i] = newpt
            simpview[inext] = newpt

            push!(new_simps, newsimp)
        end

        ncut = 0
        for simplex in eachcol(simplices)
            if all(isfree[simplex])
                edges = [
                    begin
                        inext = i + 1
                        if inext > length(simplex)
                            inext = 1
                        end

                        (i, inext)
                    end for i = 1:length(simplex)
                ]
                lengths = map(
                    t -> let (i, inext) = t
                        norm(points[:, simplex[inext]] .- points[:, simplex[i]])
                    end,
                    edges
                )

                imax = argmax(lengths)
                L = lengths[imax]

                # split it!
                if L > Lmax
                    split_simplex!(simplex, edges[imax]...)
                end
            end
        end

        if length(new_simps) == 0
            return (
                Stereolitography(points, simplices), length(new_simps)
            )
        end

        points = hcat(
            points, reduce(hcat, new_pts)
        )
        simplices = hcat(
            simplices, reduce(hcat, new_simps)
        )

        (
            Stereolitography(points, simplices), length(new_simps)
        )

    end

    """
    $TYPEDSIGNATURES

    Refine a stereolitography object until it matches a given edge length.
    """
    function refine_to_length(stl::Stereolitography, Lmax::Float64)

        nnew = 1
        while nnew > 0
            stl, nnew = _cut_larger(stl, Lmax)
        end

        stl

    end

    _exponents(nd::Int64) = Iterators.product(
        fill((0, 1), nd)...
    ) |> collect |> vec

    """
    Convert coordinates to bilinear basis
    """
    to_bilinear(coords::AbstractVector{Float64}) = let exps = _exponents(length(coords))
        map(e -> prod(coords .^ e), exps)
    end
    to_bilinear(coords::AbstractMatrix{Float64}) = mapslices(to_bilinear, coords; dims = 1)

    """
    Obtain corners of a hypercube
    """
    function hypercube_corners(origin::Vector{Float64}, widths::Vector{Float64})
        d = length(origin)  # Dimensionality of the hypercube
        n_corners = 2^d     # Number of corners in a hypercube

        # Initialize the matrix to store the corner coordinates
        pts = zeros(Float64, d, n_corners)

        # Generate all combinations of corners
        for i in 0:(n_corners-1)
            for j in 1:d
                # Determine if the j-th dimension is at the origin or origin + width
                pts[j, i+1] = origin[j] + ((i >> (j-1)) & 1) * widths[j]
            end
        end

        return pts
    end

    """
    Make linear regressions for the projection and signed distance to a surface
    as a function of spatial position. Returned in matrix and vector form, respectively.
    Uses corners as exact data points.
    Also returns boolean indicating whether additional refinement is needed.
    """
    function proj_distance_regression(
            tree::STLTree, origin::Vector{Float64}, widths::Vector{Float64};
            outside_reference = nothing,
            atol::Float64, rtol::Float64,
    )
        pts = hypercube_corners(origin, widths)
        X = to_bilinear(pts)

        projs = similar(pts)
        dists = Vector{Float64}(undef, size(pts, 2))

        for (j, c) in enumerate(eachcol(pts))
            p, d = tree(c)
            d *= (1 - 2 * point_in_polygon(tree, c; outside_reference = outside_reference))

            projs[:, j] .= p
            dists[j] = d
        end

        Xinv = pinv(X')

        Aproj = Xinv * projs' |> permutedims
        Adist = Xinv * dists

        dist_dev = 0.0
        dmin = Inf
        center = origin .+ widths ./ 2
        for pt in eachcol(pts)
            test_pt = @. (center + pt) / 2

            _, d = tree(test_pt)
            d *= (1 - 2 * point_in_polygon(tree, test_pt; outside_reference = outside_reference))

            dhat = Adist ⋅ to_bilinear(test_pt)

            dist_dev = max(abs(dhat - d), dist_dev)
            dmin = min(dmin, abs(d))
        end

        refine = dist_dev > max(0.0, dmin - norm(widths) / 2) * rtol + atol

        (Aproj, Adist, refine)
    end

    """
    $TYPEDFIELDS

    Struct to describe an approximate distance field via Barnes-Hut tree
    """
    struct DistanceField
        Aproj::Union{Matrix{Float64}, Nothing}
        Adist::Union{Vector{Float64}, Nothing}
        left_field::Union{DistanceField, Nothing}
        right_field::Union{DistanceField, Nothing}
        split_dim::Int64
        split_point::Float64
    end

    """
    $TYPEDSIGNATURES

    Constructor for an approximate distance field via Barnes-Hut tree.

    Splitting of the Barnes-Hutt tree is stopped once the deviations in signed distance function
    predictions respect relative and absolute tolerances `atol, rtol`.
    """
    function DistanceField(
        tree_or_stl::Union{STLTree, Stereolitography}, origin::Vector{Float64}, widths::Vector{Float64};
        outside_reference = nothing, 
        atol::Float64 = 1e-3,
        rtol::Float64 = 0.5,
        _depth::Int64 = 1
    )
        tree = (
            tree_or_stl isa STLTree ?
            tree_or_stl :
            STLTree(tree_or_stl)
        )

        Aproj, Adist, refine = proj_distance_regression(tree, origin, widths; 
                                                        atol = atol, 
                                                        rtol = rtol,
                                                        outside_reference = outside_reference)

        if refine # split!
            if _depth > length(origin)
                _depth = 1
            end

            new_widths = copy(widths)
            new_widths[_depth] /= 2
            
            left_origin = copy(origin)
            right_origin = copy(origin)
            right_origin[_depth] += new_widths[_depth]

            return DistanceField(
                                 nothing, nothing,
                                 DistanceField(tree, left_origin, new_widths; atol = atol, rtol = rtol, 
                                               outside_reference = outside_reference, _depth = _depth + 1),
                                 DistanceField(tree, right_origin, new_widths; atol = atol, rtol = rtol, 
                                               outside_reference = outside_reference, _depth = _depth + 1),
                                 _depth, right_origin[_depth]
            )
        end

        DistanceField(Aproj, Adist,
                     nothing, nothing,
                    0, 0.0)
    end

    """
    $TYPEDSIGNATURES

    Find projection to a surface and distance using an approximate distance field
    """
    function (field::DistanceField)(x::AbstractVector)

        f = field
        while f.split_dim != 0 # not a leaf
            if x[f.split_dim] < f.split_point
                f = f.left_field
            else
                f = f.right_field
            end
        end

        ξ = to_bilinear(x)
        p = f.Aproj * ξ

        (p, norm(p .- x))

    end

    """
    $TYPEDSIGNATURES

    Point-in-polygon query using a distance field.
    Note that the external reference must be used passed when the distance field 
    is created. Kwargs are dummies.
    """
    function point_in_polygon(field::DistanceField, x::AbstractVector; kwargs...)

        f = field
        while f.split_dim != 0 # not a leaf
            if x[f.split_dim] < f.split_point
                f = f.left_field
            else
                f = f.right_field
            end
        end

        ξ = to_bilinear(x)
        sdf = f.Adist ⋅ ξ

        (sdf < 0.0)

    end

end
