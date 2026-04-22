module BlockMesher

    using DocStringExtensions

    using LinearAlgebra
    using NearestNeighbors

    using DelimitedFiles
    using WriteVTK

    export Stereolitography, refine_to_length, merge_points,
        Box, Ball, Line, DistanceField,
        feature_regions, centers_and_normals,
        vtk_grid, vtk_save,
        Mesh, get_cells

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
            0.0f0,
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
            vertices = Vector{Vector{Float32}}()
            faces = Vector{Vector{Int64}}()

            face = Int64[]
            open(filename, "r") do file
                for _line in eachline(file)
                    line = strip(_line)

                    if startswith(line, "vertex")
                        # Extract vertex coordinates
                        coords = split(line)
                        x = parse(Float32, coords[2])
                        y = parse(Float32, coords[3])
                        z = parse(Float32, coords[4])
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

                points = zeros(Float32, 3, 3 * ntri)
                simplices = zeros(Int64, 3, ntri)

                for k = 1:ntri
                    _ = popN(12) # normal

                    points[:, 3*(k-1)+1] .= (reinterpret(Float32, popN(12)))
                    points[:, 3*(k-1)+2] .= (reinterpret(Float32, popN(12)))
                    points[:, 3*(k-1)+3] .= (reinterpret(Float32, popN(12)))

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
        points::AbstractMatrix
        simplices::AbstractMatrix
    end

    """
    $TYPEDSIGNATURES

    Obtain stereolitography object from an array of points in Selig format
    (counter-clockwise, forming a 2D surface, with each column representing a point).
    If `closed = true` (default), a closed surface is imposed.
    """
    function Stereolitography(
        points::AbstractMatrix;
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
            return Stereolitography(
                permutedims(readdlm(fname)) |> x -> Float32.(x); 
                closed = true
            )
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
        new_points = AbstractVector[]

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
        simplex::Matrix, h::Real;
        growth_ratio::Real = 1.1,
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
        growth_ratio::Real = 1.1,
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
        simplex::AbstractMatrix
    ) = [
        let idxs = setdiff(1:size(simplex, 2), i)
            simplex[:, idxs]
        end for i = 1:size(simplex, 2)
    ]

    """
    Obtain projection of a point upon a simplex
    """
    function proj2simplex(
        simplex::AbstractMatrix, pt::AbstractVector
    )
        ϵ = 1f-14 # eps(eltype(simplex))

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
            d = Inf32

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
    function _simplex_normal(simplex::AbstractMatrix, normalize::Bool = true)

        ϵ = 1f-14 # eps(eltype(simplex))

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

    _simplex_center(simplex::AbstractMatrix) = dropdims(
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

    _face2tag(f::AbstractVector) = sort(f) |> f -> tuple(f...)

    """
    $TYPEDSIGNATURES

    Find simplices that violate minimum radius and maximum normal angle criteria
    and return them in a new STL object
    """
    function feature_regions(stl::Stereolitography;
        angle::Real = 15.0,
        radius::Real = Inf64,
        include_boundaries::Bool = false)
        ϵ = eps(Float32)
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
        centers::AbstractMatrix
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
    (dist::DistanceField)(x::AbstractVector) = nn(
        dist.tree, x
    )[2]

    """
    $TYPEDSIGNATURES

    Obtain projection upon surface, given distance field.
    Searches for simplices with centers within a given range
    as candidates for projection.
    """
    function projection(
        dist::DistanceField, x::AbstractVector, R::Real = 0.0
    )
        idx, d = nn(dist.tree, x)
        p = dist.centers[:, idx]

        if R > d
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
        origin::Vector{Float32}, widths::Vector{Float32},
        growth_ratio::Real = 1.1
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
    $TYPEDSIGNATURES

    (Orderly) refine STLs to distance fields and local refinement levels.
    Returns the corresponding `DistanceField` structs.
    
    Inputs are tuples or pairs between stereolitography objects and the local 
    refinement.

    Refinement regions are passed as tuples between distance functions and
    local refinement.

    `ratio` is a multiplying factor applied to all refinement levels.
    """
    function refine_orderly(
        surfaces...;
        refinement_regions::AbstractVector = [],
        ratio::Real = 0.5f0,
        growth_ratio::Real = 2.0f0,
        tolerance::Real = 1f-7,
    )
        surface_order = let hs = map(
            s -> s[2], surfaces |> collect
        )
            sortperm(hs)
        end
        result_dict = Dict{Int64, DistanceField}()

        refinement_regions = Any[
            (t[1], t[2] * ratio) for t in refinement_regions
        ]

        for i in surface_order
            stl, h = surfaces[i]
            h *= ratio

            stl = refine_to_length(
                stl, h;
                tolerance = tolerance,
                refinement_regions = refinement_regions,
                growth_ratio = growth_ratio
            )

            dfield = DistanceField(stl)
            result_dict[i] = dfield

            push!(
                refinement_regions, (dfield, h)
            )
        end

        map(
            i -> result_dict[i], 1:length(surfaces)
        )
    end

    """
    $TYPEDFIELDS

    Struct to define a mesh.
    Matrices are of shape `(ndims, nblocks)`.
    """
    struct Mesh
        origin::AbstractVector{Float32}
        widths::AbstractVector{Float32}
        block_size::Int32
        block_origins::AbstractMatrix{Float32}
        block_widths::AbstractMatrix{Float32}
        distance_fields::Dict{String, DistanceField}
    end

    """
    $TYPEDSIGNATURES

    Generate a mesh given hypercube origins and widths.

    Surfaces should be specified as tuples with family names,
    `Stereolitography` structs and local refinement levels, respectively.

    Refinement regions, meanwhile, may be specified as tuples between distance
    functions (see `Line, Box, Ball, DistanceField` in this module) and local refinement levels.

    An approximate growth rate is accepted for the block octree/quadtree.
    The block size (along all axes) is given by `block_size` (def. 8).

    Example:

    ```
    stl = Stereolitography("wall.dat") # or STL in 3D
    stl2 = Stereolitography("wall2.dat")

    features = feature_regions(stl) |> DistanceField
    region2 = Stereolitography("region.stl") |> DistanceField

    msh = Mesh(
        [-1.0, -1.0], [3.0, 3.0], # origin, widths
        ("wall", stl, 1e-3),
        ("wall2", stl2, 2e-3);
        growth_ratio = 2.0f0, # default
        refinement_regions = [
            features => 5e-4,
            region2 => 1e-2,
            Ball([0.0, 0.0], 1.0) => 2e-2,
            Box([-1.0, -1.0], [1.0, 2.0]) => 1e-2
        ],
    )
    ```
    """
    function Mesh(
        origin::AbstractVector, widths::AbstractVector,
        surfaces...;
        growth_ratio::Real = 2.0f0,
        tolerance::Real = 1f-7,
        block_size::Int = 8,
        refinement_regions::AbstractVector = [],
        verbose::Bool = false,
    )
        verbose && println("==Starting block-mesh generation procedure...==")

        t0 = time()
        verbose && println("Refining surfaces to local refinement levels...")
        
        block_size = Int32(block_size)

        origin = Float32.(origin)
        widths = Float32.(widths)

        hs = Dict(
            [
                sname => h for (sname, _, h) in surfaces
            ]...
        )
        dfields = let dfields = refine_orderly(
            [
                (stl, h) for (_, stl, h) in surfaces
            ]...;
            refinement_regions = refinement_regions,
            growth_ratio = growth_ratio, tolerance = tolerance
        )
            Dict(
                [
                    t[1] => dfield for (t, dfield) in zip(
                        surfaces, dfields
                    )
                ]...
            )
        end
        verbose && println("[DONE] - $(time() - t0) seconds elapsed")

        ref_regions = Any[
            (t[1], t[2] * block_size) for t in refinement_regions
        ]
        for sname in keys(dfields)
            push!(
                ref_regions,
                (dfields[sname], hs[sname] * block_size)
            )
        end

        verbose && println("Refining region tree...")
        t0 = time()

        block_origins, block_widths = let ows = refine_octree(
            ref_regions, origin, widths,
            growth_ratio
        )
            (
                map(t -> t[1], ows) |> x -> reduce(hcat, x),
                map(t -> t[2], ows) |> x -> reduce(hcat, x),
            )
        end

        verbose && println("[DONE] - $(time() - t0) seconds elapsed")

        verbose && println("==Done with block-mesh generation procedure!==")

        Mesh(
            origin, widths,
            block_size,
            block_origins, block_widths,
            dfields,
        )
    end

    _range_prod(ranges::Union{AbstractVector, AbstractRange}...) = Base.Iterators.product(
        ranges...
    ) |> collect |> vec
    _range_prod(range::Union{AbstractVector, AbstractRange}, N::Int) = _range_prod(
        fill(range, N)...
    )

    """
    $TYPEDSIGNATURES

    Obtain all points in a given range of partitions of a mesh
    (default: all). 

    Returns cell centers, widths and boolean masks indicating whether the cell
    is in the margin of a block. Centers and widths have shape `(ndims, npts)`.
    """
    function get_cells(
        msh::Mesh, range = nothing;
        margin::Int = 0
    )
        if isnothing(range)
            range = 1:size(msh.block_widths, 2)
        end

        block_origins = msh.block_origins[:, range]
        block_widths = msh.block_widths[:, range]

        margin = Int32(margin)
        nd = size(block_origins, 1)
        n_per_block = (msh.block_size + 2 * margin) ^ nd

        inner_coordinates = _range_prod(
            (
                (0.5f0 - margin):1.0f0:(msh.block_size + margin - 0.5f0)
            ) ./ msh.block_size, nd
        ) |> x -> collect.(x) |> x -> reduce(hcat, x)

        centers = map(
            (o, w) -> inner_coordinates .* w .+ o,
            eachcol(block_origins), eachcol(block_widths)
        ) |> x -> reduce(hcat, x)
        widths = repeat(
            block_widths ./ msh.block_size;
            inner = (1, n_per_block)
        )

        is_margin = let im = trues(n_per_block)
            k = 0
            for idxs in _range_prod(
                (1 - margin):(msh.block_size + margin), nd
            )
                k += 1

                if all(idxs .>= 1) && all(
                    idxs .<= msh.block_size
                )
                    im[k] = false
                end
            end

            repeat(im; outer = size(block_origins, 2))
        end

        (centers, widths, is_margin)
    end

    _fix_export(v::AbstractVector) = v
    _fix_export(a::AbstractArray) = (
        size(a, ndims(a)) > size(a, 1) ?
        a :
        let s = 1:ndims(a) |> collect
            circshift!(s, -1)

            permutedims(a, tuple(s...))
        end
    )

    _sellast(v::AbstractArray, i) = selectdim(
        v, ndims(v), i
    ) |> copy

    """
    $TYPEDSIGNATURES

    Create folder with name `fname` with multi-block VTK file.
    kwargs are exported as volume data.

    Only a given set of partitions may be exported if indices `partition_indices`
        are specified.
    """
    function WriteVTK.vtk_grid(
        fname::String, msh::Mesh,
        partition_indices = nothing; 
        _make_folder::Bool = true,
        kwargs...
    )
        nd = size(msh.block_origins, 1)
        nblocks = size(msh.block_origins, 2)
        nperblock = msh.block_size ^ nd

        if isnothing(partition_indices)
            partition_indices = 1:nblocks
        end

        if _make_folder
            if isdir(fname)
                @warn "Overwriting volume output in folder $fname."
                rm(fname; recursive = true, force = true)
            end
            mkdir(fname)
        end

        vtm = joinpath(fname, "VOLUME") |> vtk_multiblock

        for block in partition_indices
            o = msh.block_origins[:, block]
            w = msh.block_widths[:, block]

            grid = vtk_grid(
                vtm, [
                    LinRange(o[dim], o[dim] + w[dim], msh.block_size + 1) for dim = 1:nd
                ]...
            )

            block_range = (
                (block - 1) * nperblock + 1
            ): (
                nperblock * block
            )

            for (k, v) in kwargs
                v = _fix_export(v)
                grid[String(k)] = _sellast(v, block_range)
            end
        end

        vtm
    end

end
