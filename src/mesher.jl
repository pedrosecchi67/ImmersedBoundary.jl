module Mesher

    using DocStringExtensions

    using LinearAlgebra
    using NearestNeighbors

    using DelimitedFiles
    using WriteVTK

    export Stereolitography, refine_to_length, merge_points,
        Box, Ball, Line, DistanceField,
        feature_regions, centers_and_normals,
        vtk_grid, vtk_save,
        Mesh

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

    Obtain stereolitography object from an array of points in Selig format
    (counter-clockwise, forming a 2D surface, with each column representing a point).
    If `closed = true` (default), a closed surface is imposed.
    """
    function Stereolitography(
        points::AbstractMatrix{Float64};
        closed::Bool = true,
    )
        simplices = let inds = 1:size(points, 2)
            (
                closed ?
                [
                    inds'; circshift(inds, -1)'
                ] :
                [
                    inds[1:(end - 1)]'; inds[2:end]'
                ]
            )
        end

        Stereolitography(points, simplices)
    end

    """
    $TYPEDSIGNATURES

    Obtain stereolitography data from mesh file.

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

    Obtain VTK grid from stereolitography object.
    Kwargs are saved as point/cell data
    """
    function WriteVTK.vtk_grid(
        fname::String, stl::Stereolitography, vtm_file = nothing;
        kwargs...
    )
        points = stl.points
        nd = size(points, 1)
        cells = [
            MeshCell(
                (
                    nd == 2 ?
                    WriteVTK.VTKCellTypes.VTK_LINE :
                    WriteVTK.VTKCellTypes.VTK_TRIANGLE
                ), copy(simp)
            ) for simp in eachcol(stl.simplices)
        ]

        vtk = nothing
        if isnothing(vtm_file)
            vtk = vtk_grid(
                fname, points, cells
            )
        else
            vtk = vtk_grid(
                vtm_file, fname, points, cells
            )
        end

        for (k, v) in kwargs
            if v isa AbstractArray
                if size(v, ndims(v)) == length(cells) || size(v, ndims(v)) == size(points, 2)
                    vtk[String(k)] = v
                else
                    vtk[String(k)] = permutedims(v)
                end
            else
                vtk[String(k)] = v
            end
        end

        vtk
    end

    """
    $TYPEDSIGNATURES

    Merge two (or more) stereolitography objects together according to tolerance
    """
    function merge_points(
        stls::Stereolitography...;
        tolerance::Real = 1e-7,
        clean_degenerate::Bool = true,
    )
        pt2tag = pt -> tuple(
            (
                Int64.(round.(pt ./ tolerance))
            )...
        )
        new_points = Vector{Float64}[]

        nd = size(stls[1].points, 1)
        tag2ind = Dict{
            NTuple{nd, Int64}, Int64
        }()
        N = 0

        get_index! = pt -> let tag = pt2tag(pt)
            if haskey(tag2ind, tag)
                return tag2ind[tag]
            end

            N += 1
            tag2ind[tag] = N
            push!(new_points, pt)

            N
        end

        new_simplices = Matrix{Int64}[]
        for stl in stls
            new_indices = map(
                get_index!, eachcol(stl.points)
            )

            push!(
                new_simplices, 
                new_indices[stl.simplices]
            )
        end

        new_points = reduce(hcat, new_points)
        new_simplices = reduce(hcat, new_simplices)

        if clean_degenerate
            mask = map(
                simp -> let nuniq = unique(simp) |> length
                    length(simp) == nuniq
                end, eachcol(new_simplices)
            )

            new_simplices = new_simplices[:, mask]
        end

        Stereolitography(new_points, new_simplices)
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
        let n = 0
            mapreduce(
                s -> begin
                    nold = n
                    n += size(s.points, 2)

                    s.simplices .+ nold
                end, hcat, stl
            )
        end,
    )

    """
    Recursive function to split triangle until max. length is met.
    `refinement_regions` may contain distance-function/local refinement
    tuples for added rules.
    """
    function refine_to_length!(
        simplex::Matrix{Float64}, h::Real;
        growth_ratio::Float64 = 1.1,
        refinement_regions::AbstractVector = []
    )
        max_violation = 0.0
        index = 0

        get_next = i -> (
            i == size(simplex, 2) ? 1 : i + 1
        )

        for i = 1:size(simplex, 2)
            inext = get_next(i)

            p1 = simplex[:, i]
            p2 = simplex[:, inext]
            phalf = @. (p1 + p2) / 2

            L = norm(p2 .- p1)

            hloc = h
            for (df, href) in refinement_regions
                hloc = min(
                    hloc, max((df(phalf) - L) * (growth_ratio - 1.0), href)
                )
            end

            violation = L - hloc
            if max_violation < violation
                max_violation = violation
                index = i
            end
        end

        if index == 0
            return [simplex]
        end

        inext = get_next(index)

        p1 = @view simplex[:, index]
        p2 = @view simplex[:, inext]
        pnew = @. (p1 + p2) / 2

        new_simplex = copy(simplex)
        p2 .= pnew
        new_simplex[:, index] .= pnew

        [
            refine_to_length!(simplex, h; 
                refinement_regions = refinement_regions,
                growth_ratio = growth_ratio); 
            refine_to_length!(new_simplex, h; 
                refinement_regions = refinement_regions,
                growth_ratio = growth_ratio); 
        ]
    end

    """
    $TYPEDSIGNATURES

    Split triangles in a stereolitography object until all edges are at most equal to
    a given length
    """
    function refine_to_length(
        stl::Stereolitography, h::Real;
        tolerance::Real = 1e-7,
        growth_ratio::Float64 = 1.1,
        refinement_regions::AbstractVector = []
    )
        stl = begin
            points = map(
                simp -> refine_to_length!(
                    stl.points[:, simp], h;
                    refinement_regions = refinement_regions,
                    growth_ratio = growth_ratio,
                ), eachcol(stl.simplices)
            ) |> x -> reduce(vcat, x)
            nd = size(first(points), 2)
            points = reduce(hcat, points)

            simplices = reshape(
                collect(1:size(points, 2)), nd, :
            )

            Stereolitography(points, simplices)
        end

        merge_points(stl; tolerance = tolerance)
    end

    """
    Obtain faces of simplices
    """
    simplex_faces(
        simplex::Matrix{Float64}
    ) = [
        let idxs = setdiff(1:size(simplex, 2), i)
            simplex[:, idxs]
        end for i = 1:size(simplex, 2)
    ]

    """
    Obtain projection of a point upon a simplex
    """
    function proj2simplex(
        simplex::Matrix{Float64}, pt::AbstractVector{Float64}
    )
        ϵ = eps(Float64)

        if size(simplex, 2) == 1
            return vec(simplex)
        elseif size(simplex, 2) == 2
            p0 = simplex[:, 1]
            p1 = simplex[:, 2]
            u = p1 .- p0

            ξ = (
                (pt .- p0) ⋅ u
            ) / (u ⋅ u + ϵ)

            if ξ < - ϵ
                return p0
            elseif ξ > 1.0 + ϵ
                return p1
            else
                return p0 .+ u .* ξ
            end
        end

        p0 = simplex[:, 1]
        M = simplex[:, 2:end] .- p0

        ξ = pinv(M) * (pt .- p0)

        if any(
            xi -> xi < - ϵ, ξ
        ) || (
            sum(ξ) > 1.0 + ϵ
        )
            p = similar(pt)
            d = Inf64

            for face in simplex_faces(simplex)
                _p = proj2simplex(face, pt)
                _d = norm(_p .- pt)

                if _d < d
                    d = _d
                    p .= _p
                end
            end

            return p
        end

        p0 .+ M * ξ
    end

    """
    Get simplex normal
    """
    function _simplex_normal(simplex::AbstractMatrix{Float64}, normalize::Bool = true)

        ϵ = eps(eltype(simplex))

        if size(simplex, 1) == 2
            v = simplex[:, 2] .- simplex[:, 1]

            n = [
                v[2], - v[1]
            ]
            if normalize
                return n ./ (norm(v) + ϵ)
            else
                return n
            end
        end

        p0 = simplex[:, 1]

        n = cross(simplex[:, 2] .- p0, simplex[:, 3] .- p0)

        if normalize
            return n ./ (norm(n) + ϵ)
        end

        n

    end

    _simplex_center(simplex::Matrix{Float64}) = dropdims(
        sum(simplex; dims = 2); dims = 2
    ) ./ size(simplex, 2)

    """
    $TYPEDSIGNATURES

    Obtain simplex centers and normals (with norms equal to simplex areas).
    """
    function centers_and_normals(stl::Stereolitography)

        simplices = map(
            simp -> stl.points[:, simp], eachcol(stl.simplices)
        )

        centers = reduce(
            hcat,
            map(
                _simplex_center, simplices
            )
        )
        normals = reduce(
            hcat,
            map(
                s -> _simplex_normal(s, false), simplices
            )
        )

        (centers, normals)

    end

    _face2tag(f::AbstractVector{Int64}) = sort(f) |> f -> tuple(f...)

    """
    $TYPEDSIGNATURES

    Find simplices that violate minimum radius and maximum normal angle criteria
    and return them in a new STL object
    """
    function feature_regions(stl::Stereolitography;
        angle::Float64 = 15.0,
        radius::Float64 = Inf64,
        include_boundaries::Bool = false)
        ϵ = eps(Float64)
        nd = size(stl.points, 1)

        T = NTuple{nd - 1, Int64}

        angle = deg2rad(max(angle, 1.0))
        max_cos = cosd(0.05)

        edges = Tuple{Int64, Int64}[]
        registry = Dict{T, Int64}()
        for (i, simp) in eachcol(stl.simplices) |> enumerate
            for pivot in simp
                face = _face2tag(
                    setdiff(simp, pivot)
                )

                if haskey(registry, face)
                    push!(
                        edges, (registry[face], i)
                    )
                    delete!(registry, face)
                else
                    registry[face] = i
                end
            end
        end

        # add remaning (border) faces too
        for (_, ind) in registry
            push!(edges, (ind, ind))
        end

        centers, normals = centers_and_normals(stl)

        included_simplices = falses(size(centers, 2))
        for (i, j) in edges
            ni = normals[:, i]
            nj = normals[:, j]

            ni ./= (norm(ni) + ϵ)
            nj ./= (norm(nj) + ϵ)

            θ = acos(
                min(ni ⋅ nj, max_cos)
            )
            d = norm(centers[:, i] .- centers[:, j])

            if (i == j && include_boundaries) || (d / θ < radius) || (θ > angle)
                included_simplices[i] = true
                included_simplices[j] = true
            end
        end

        Stereolitography(stl.points, stl.simplices[:, included_simplices])
    end

    """
    $TYPEDFIELDS

    Struct to describe an approximate distance field around a stereolitography
    object.
    """
    struct DistanceField
        stl::Stereolitography
        centers::Matrix{Float64}
        tree::KDTree
    end

    """
    $TYPEDSIGNATURES

    Constructor for a distance field.
    Refinement is applied if max. edge size `h` is provided
    """
    function DistanceField(
        stl::Stereolitography;
        leaf_size::Int = 25, h::Real = 0.0
    )
        if h > 0.0
            stl = refine_to_length(stl, h)
        end

        centers, _ = centers_and_normals(stl)
        tree = KDTree(centers; leafsize = leaf_size)

        DistanceField(stl, centers, tree)
    end

    """
    $TYPEDSIGNATURES

    Obtain approximate distance from distance field
    """
    (dist::DistanceField)(x::AbstractVector{Float64}) = nn(
        dist.tree, x
    )[2]

    """
    $TYPEDSIGNATURES

    Obtain projection upon surface, given distance field.
    Searches for simplices with centers within a given range
    as candidates for projection.
    """
    function projection(
        dist::DistanceField, x::AbstractVector{Float64}, R::Real = 0.0
    )
        idx, d = nn(dist.tree, x)
        p = dist.centers[:, idx]

        if R > 0.0
            idxs = inrange(dist.tree, x, R)

            for i in idxs
                simp = dist.stl.points[:, dist.stl.simplices[:, i]]

                _p = proj2simplex(simp, x)
                _d = norm(_p .- x)

                if _d < d
                    d = _d
                    p .= _p
                end
            end
        end

        p
    end

    """
    $TYPEDSIGNATURES

    Refine a cell until all refinement criteria (distance function/size tuples)
    are met. Includes global growth ratio.
    Returns vector of tuples, each tuple with an origin vector and a widths vector.
    The closest possible thing to isotropy is sought.
    """
    function refine_octree(
        refinement_criteria::AbstractVector,
        origin::Vector{Float64}, widths::Vector{Float64},
        growth_ratio::Float64 = 1.1
    )
        L = maximum(widths)
        R = norm(widths) / 2 # circumradius
        center = @. origin + widths / 2

        criteria_is_active = map(
            t -> let (df, h) = t
                Lmax = max(
                    (growth_ratio - 1.0) * (
                        df(center) - R
                    ), h
                )

                (Lmax < L)
            end, refinement_criteria
        )

        if !any(criteria_is_active)
            return [(origin, widths)]
        end

        refinement_criteria = refinement_criteria[criteria_is_active]

        split_sizes = let wmin = minimum(widths)
            Int64.(
                round.(widths ./ wmin)
            ) .+ 1
        end

        new_widths = widths ./ split_sizes
        new_origins = Iterators.product(
            map(
                (o, w, s) -> LinRange(o, o + w, s + 1)[1:(end - 1)],
                origin, widths, split_sizes
            )...
        ) |> collect |> vec

        reduce(
            vcat,
            map(
                o -> refine_octree(
                    refinement_criteria,
                    collect(o), new_widths,
                    growth_ratio
                ), new_origins
            )
        )
    end

    """
    Run DFS on graph.
    """
    function dfs(graph::Vector{Vector{Int64}}, start_node::Int64)::Vector{Int64}
        # Initialize visited array
        visited = falses(length(graph))
        
        # Initialize result vector
        result = Int64[]
        
        # Use a stack for iterative DFS
        stack = Set([start_node])
        
        while !isempty(stack)
            # Pop the top node from the stack
            node = pop!(stack)
            
            if !visited[node]
                # Mark as visited and add to result
                visited[node] = true
                push!(result, node)
                
                # Push all unvisited neighbors to the stack
                for neighbor in graph[node]
                    if !visited[neighbor]
                        push!(stack, neighbor)
                    end
                end
            end
        end
        
        return result
    end

    """
    $TYPEDFIELDS

    Struct to define a mesh
    """
    struct Mesh
        origins::Matrix{Float64}
        widths::Matrix{Float64}
        centers::Matrix{Float64}
        in_domain::Vector{Bool}
        family_distances::Dict{String, Vector{Float64}}
        family_projections::Dict{String, Matrix{Float64}}
        families::Dict{String, Stereolitography}
    end

    """
    $TYPEDSIGNATURES

    Generate an octree/quadtree mesh described by:

        * A hypercube origin;
        * A vector of hypercube widths;
        * A set of tuples in format `(name, surface, max_length)` describing
            stereolitography surfaces (`Mesher.Stereolitography`) and 
            the max. cell widths at these surfaces;
        * A set of refinement regions described by distance functions and
            the local refinement at each region. Example:
                ```
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

    ```
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
    """
    function Mesh(
        origin::Vector{Float64}, widths::Vector{Float64},
        surfaces::Tuple{String, Stereolitography, Float64}...;
        interior_reference::Union{Vector{Float64}, Nothing} = nothing,
        growth_ratio::Float64 = 1.1,
        ghost_layer_ratio::Float64 = 1.5,
        refinement_regions = [],
        hypercube_families = [],
        merge_tolerance::Real = 1e-7,
        verbose::Bool = false,
    )
        verbose && println("====Starting mesh generation====")

        t0_global = time()

        verbose && println("Adding surface refinement to match volume regions...")

        t0 = time()
        # refine surfaces to refinement regions and to their own
        # characteristic lengths
        surfaces = [
            (
                sname,
                refine_to_length(stl, h / 2;
                    tolerance = merge_tolerance,
                    growth_ratio = growth_ratio,
                    refinement_regions = refinement_regions),
                h
            ) for (sname, stl, h) in surfaces
        ]

        # now add refinement according to the other surfaces
        let hs = map(t -> t[3], surfaces)
            asrt = sortperm(hs)

            for (k, i) in enumerate(asrt)
                stl_i = surfaces[i][2]
                h = surfaces[i][3]
                dfield = DistanceField(stl_i)

                for j in asrt[(k + 1):end]
                    sname, stl, L = surfaces[j]

                    surfaces[j] = (
                        sname,
                        refine_to_length(
                            stl, L;
                            tolerance = merge_tolerance,
                            growth_ratio = growth_ratio,
                            refinement_regions = [
                                (dfield, h / 2)
                            ]
                        ),
                        L
                    )
                end
            end
        end

        verbose && println("[DONE] $(time() - t0) seconds elapsed")

        verbose && println("Creating surface approx. distance fields...")

        t0 = time()
        # create dist. fields and add them to ref. regions
        distance_fields = DistanceField[] # for re-use
        refinement_regions = [
            refinement_regions;
            [
                let dfield = DistanceField(stl)
                    push!(distance_fields, dfield)

                    (dfield, h)
                end for (
                    _, stl, h
                ) in surfaces
            ]
        ]

        verbose && println("[DONE] $(time() - t0) seconds elapsed")

        verbose && println("Creating joint surfaces for future family representation...")

        t0 = time()
        # joining surfaces
        stl_families = Dict{String, Vector{Stereolitography}}()
        for (sname, stl, _) in surfaces
            if length(sname) > 0
                if !haskey(stl_families, sname)
                    stl_families[sname] = Stereolitography[]
                end

                push!(stl_families[sname], stl)
            end
        end

        stl_families = Dict(
            sname => cat(stls...) for (sname, stls) in stl_families
        )

        verbose && println("[DONE] $(time() - t0) seconds elapsed")

        verbose && println("Refining octree to distance fields...")

        t0 = time()
        origins, cell_widths = let cells = refine_octree(
            refinement_regions,
            origin, widths, growth_ratio
        )
            (
                map(t -> t[1], cells) |> x -> reduce(hcat, x),
                map(t -> t[2], cells) |> x -> reduce(hcat, x),
            )
        end
        centers = @. origins + cell_widths / 2
        ncells = size(centers, 2)
        nd = size(centers, 1)

        verbose && println("$ncells cells generated")

        verbose && println("[DONE] $(time() - t0) seconds elapsed")

        verbose && println("Calculating cell-family distances...")

        t0 = time()
        # calculate projs/dists using distance fields
        family_distances = Dict{String, Vector{Float64}}()
        family_projections = Dict{String, Matrix{Float64}}()

        radii = sum(
            cell_widths .^ 2; dims = 1
        ) |> vec |> x -> sqrt.(x) ./ 2

        # calculating distance to each surface family
        for ((sname, _, _), dfield) in zip(
            surfaces, distance_fields
        )
            if length(sname) > 0 # not only meshing resource
                if !haskey(family_distances, sname)
                    family_distances[sname] = fill(Inf64, ncells)
                    family_projections[sname] = Matrix{Float64}(undef, nd, ncells)
                end

                dists = family_distances[sname]
                projs = family_projections[sname]

                for (i, c) in eachcol(centers) |> enumerate
                    p = projection(
                        dfield, c, ghost_layer_ratio * 2 * radii[i] # calculate with precision if ghost
                    )
                    d = norm(c .- p)

                    if d < dists[i]
                        dists[i] = d
                        projs[:, i] .= p
                    end
                end
            end
        end

        # establish hypercube families
        for (family, faces) in hypercube_families
            ps = similar(centers)
            ds = fill(Inf64, size(centers, 2))

            for (dim, front) in faces
                projs = copy(centers)
                projs[dim, :] .= (
                    front ?
                    origin[dim] + widths[dim] :
                    origin[dim]
                )

                for (i, p) in eachcol(projs) |> enumerate
                    d = norm(p .- centers[:, i])

                    if d < ds[i]
                        ds[i] = d
                        ps[:, i] .= p
                    end
                end
            end

            family_distances[family] = ds
            family_projections[family] = ps
        end

        verbose && println("[DONE] $(time() - t0) seconds elapsed")

        # deallocate no-longer-needed variables
        surfaces = nothing
        distance_fields = nothing
        refinement_regions = nothing

        # default
        in_domain = trues(ncells)
        ϵ = eps(Float64)

        if !isnothing(interior_reference)
            verbose && println("Generating KD-tree...")
            t0 = time()

            tree = KDTree(centers)

            verbose && println("[DONE] $(time() - t0) seconds elapsed")

            t0 = time()
            verbose && println("Generating conn. graph...")

            graph = map(
                (c, r) -> inrange(tree, c, 2.1 * r),
                eachcol(centers), radii
            )
            
            # ensure reciprocity
            for (i, neighs) in enumerate(graph)
                for j in neighs
                    if i != j
                        jneighs = graph[j]

                        if !(i in jneighs)
                            push!(jneighs, i)
                        end
                    end
                end
            end
            for i = length(graph):-1:1
                neighs = graph[i]

                for j in neighs
                    if i != j
                        jneighs = graph[j]

                        if !(i in jneighs)
                            push!(jneighs, i)
                        end
                    end
                end
            end

            # removing connectivity when crossing boundaries
            for (family, projs) in family_projections
                dists = family_distances[family]

                for (i, neighs) in enumerate(graph)
                    c = centers[:, i]
                    p = projs[:, i]
                    d = dists[i]

                    normal = c .- p
                    normal ./= (
                        norm(normal) + ϵ
                    )

                    graph[i] = filter(
                        j -> let cj = centers[:, j]
                            d + (cj .- c) ⋅ normal > sqrt(ϵ)
                        end, neighs
                    )
                end
            end

            # making sure that only reciprocal links are found
            for i = 1:length(graph)
                graph[i] = filter(
                    j -> (i == j) || (i in graph[j]), graph[i]
                )
            end
            for i = length(graph):-1:1
                graph[i] = filter(
                    j -> (i == j) || (i in graph[j]), graph[i]
                )
            end

            verbose && println("[DONE] $(time() - t0) seconds elapsed")

            verbose && println("Running DFS...")
            t0 = time()

            start_node = argmin(
                sum(
                    (centers .- interior_reference) .^ 2; dims = 1
                ) |> vec
            )

            is_wet = dfs(graph, start_node)

            in_domain .= false
            in_domain[is_wet] .= true

            verbose && println("[DONE] $(time() - t0) seconds elapsed")
        end

        verbose && println("Filtering interior and ghost cells...")
        t0 = time()

        let mask = falses(ncells)
            @. mask = mask || in_domain

            for (_, dists) in family_distances
                is_ghost = @. dists < ghost_layer_ratio * radii * 2
                @. mask = mask || is_ghost
            end

            mask = findall(mask)

            origins = origins[:, mask]
            centers = centers[:, mask]
            cell_widths = cell_widths[:, mask]
            in_domain = in_domain[mask]
            family_distances = Dict(
                sname => dists[mask] for (sname, dists) in family_distances
            )
            family_projections = Dict(
                sname => projs[:, mask] for (sname, projs) in family_projections
            )
        end

        verbose && println("[DONE] $(time() - t0) seconds elapsed")

        verbose && println("====$(length(in_domain)) cells in $(time() - t0_global) seconds====")

        Mesh(
            origins, cell_widths, centers, in_domain,
            family_distances, family_projections,
            stl_families
        )
    end

    """
    $TYPEDSIGNATURES

    Write cells to VTK file. Kwargs are written as cell data.

    If `write_families = true`, name `fname * "_" * fam_name` is used
    for each surface family .vtu file. Saving is automatic.
    """
    function WriteVTK.vtk_grid(
        fname::String, msh::Mesh; 
        write_families::Bool = false,
        kwargs...
    )
        nd = size(msh.origins, 1)
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

        points = map(
            (o, w) -> multipliers .* w .+ o,
            eachcol(msh.origins), eachcol(msh.widths)
        ) |> x -> reduce(hcat, x)

        mcells = MeshCell[]
        _conn = collect(1:ncorners)
        for k = 1:size(msh.centers, 2)
            conn = _conn .+ ((k - 1) * ncorners)

            push!(
                mcells,
                MeshCell(ctype, conn)
            )
        end

        grid = vtk_grid(fname, points, mcells)
        for (k, v) in kwargs
            if v isa AbstractArray
                if size(v, ndims(v)) == length(mcells)
                    grid[String(k)] = v
                else
                    grid[String(k)] = permutedims(v)
                end
            else
                grid[String(k)] = v
            end
        end

        grid["in_domain"] = Float64.(msh.in_domain)
        for family in keys(msh.family_distances)
            grid[family * "_distances"] = msh.family_distances[family]
            grid[family * "_projections"] = msh.family_projections[family]
        end

        if write_families
            for (sname, stl) in msh.families
                vtk = vtk_grid(
                    fname * "_" * sname, stl
                )
                vtk_save(vtk)
            end
        end

        grid
    end

    """
    $TYPEDSIGNATURES

    Get mesh size
    """
    Base.length(msh::Mesh) = size(msh.centers, 2)

    """
    $TYPEDSIGNATURES

    Obtain partition of a mesh based on cell indices
    """
    Base.getindex(msh::Mesh, idxs) = Mesh(
        msh.origins[:, idxs], msh.widths[:, idxs],
        msh.centers[:, idxs], msh.in_domain[idxs],
        Dict(
            [fam => dists[idxs] for (fam, dists) in msh.family_distances]
        ),
        Dict(
            [fam => projs[:, idxs] for (fam, projs) in msh.family_projections]
        ),
        msh.families
    )

end