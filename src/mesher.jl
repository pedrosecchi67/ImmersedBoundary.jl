module Mesher

        using Base.Threads

        include("stereolitography.jl")
        using .STLHandler

        using .STLHandler.WriteVTK

        using DocStringExtensions

        using LinearAlgebra

        using JSON

        """
        $TYPEDFIELDS

        Struct to describe a projection upon a surface and its point-and-polygon status
        """
        struct Projection
                projection::Vector{Float64}
                interior::Bool
        end

        """
        $TYPEDFIELDS

        Struct to describe a cell
        """
        struct Cell
                origin::Vector{Float64}
                widths::Vector{Float64}
                projections::Vector{Projection}
        end

        """
        $TYPEDFIELDS

        Struct to overload STLTrees with naive projections upon simplices when there are 
        too few facets to justify a tree
        """
        struct NaiveSTLTree
                stl::Stereolitography
        end

        """
        $TYPEDSIGNATURES

        Obtain projection and distance on naive STL tree
        """
        function (tree::NaiveSTLTree)(x::AbstractVector{Float64})
                if size(tree.stl.simplices, 2) == 0
                    return (copy(x), 0.0)
                end

                p = STLHandler.proj2stl(tree.stl, x)
                d = norm(x .- p)

                (p, d)
        end

        """
        $TYPEDSIGNATURES

        Obtain point-in-polygon query for NaiveSTLTree struct
        """
        function STLHandler.point_in_polygon(
                tree::NaiveSTLTree, x::AbstractVector{Float64};
                outside_reference = nothing
        )
                if size(tree.stl.simplices, 2) == 0
                    return false
                end

                if isnothing(outside_reference)
                        outside_reference = map(minimum, eachrow(tree.stl.points))
                end

                STLHandler.n_crossings(tree.stl, outside_reference, x) % 2 == 1
        end

        """
        $TYPEDFIELDS

        Struct to describe a plane SDF surrogate to an STLTree
        """
        struct PlaneSDF
            normal::Vector{Float64}
            linear_coeff::Float64
        end

        """
        $TYPEDSIGNATURES

        Obtain projection and distance given PlaneSDF
        """
        function (plane::PlaneSDF)(x::AbstractVector{Float64})
                d = (x ⋅ plane.normal - plane.linear_coeff)
                p = x .- d .* plane.normal
                
                (p, d)
        end

        """
        $TYPEDSIGNATURES

        Run point-in-polygon query to PlaneSDF
        """
        function STLHandler.point_in_polygon(plane::PlaneSDF, x::AbstractVector{Float64};
                outside_reference = nothing)
                d = (x ⋅ plane.normal - plane.linear_coeff)

                if isnothing(outside_reference)
                    return (d < 0.0)
                end

                dref = (outside_reference ⋅ plane.normal - plane.linear_coeff)
                (d * dref < 0.0)
        end

        """
        $TYPEDSIGNATURES

        Obtain PlaneSDF from a projection
        """
        function PlaneSDF(p::Projection, x::AbstractVector{Float64})
                n = x .- p.projection
                nn = norm(n) + sqrt(eps(eltype(x)))
                
                if p.interior
                    nn = - nn
                end

                n ./= nn

                linear_coeff = p.projection ⋅ n
                PlaneSDF(n, linear_coeff)
        end

        """
        $TYPEDSIGNATURES

        Decide between a naive or full STL tree
        """
        function smart_stltree(stl::Stereolitography;
                leaf_size::Int64 = 1,
                threshold::Int64 = 20)
                if size(stl.simplices, 2) < threshold
                        return NaiveSTLTree(stl)
                end

                STLTree(stl; leaf_size = leaf_size)
        end

        """
        $TYPEDSIGNATURES

        Filter stereolitography object according to mask that identifies remaining
        simplices
        """
        function stereolitography_mask(stl::Stereolitography, mask)
                simplices = stl.simplices[:, mask]

                remaining_mask = falses(size(stl.points, 2))
                remaining_mask[simplices] .= true
                remaining = findall(remaining_mask)

                new_indices = zeros(Int64, size(stl.points, 2))
                new_indices[remaining] .= 1:length(remaining)

                points = stl.points[:, remaining]
                simplices = new_indices[simplices]

                Stereolitography(points, simplices)
        end

        """
        $TYPEDSIGNATURES

        Filter stereolitography to simplices within range of a given cell
        """
        function local_stereolitography(
                stl::Stereolitography, c::Cell
        )
                center = c.origin .+ c.widths ./ 2
                circumradius = norm(c.widths) / 2

                ds = map(
                        simp -> norm(
                                STLHandler.proj2simplex(stl.points[:, simp], center) .- center
                        ), 
                        eachcol(stl.simplices)
                )
                dmin = minimum(ds)

                R = dmin + circumradius * 1.1
                mask = @. ds < R

                stereolitography_mask(stl, mask)
        end

        """
        $TYPEDFIELDS

        Struct containing a triangulation and a point-in-polygon reference
        to produce Projections
        """
        struct TriReference
                triangulation::Union{STLTree, NaiveSTLTree, PlaneSDF}
                reference_point::Vector{Float64}
                interior::Bool
        end

        """
        $TYPEDSIGNATURES

        Construct a TriReference from scratch.
        """
        function TriReference(
                stl::Stereolitography;
                outside_reference = nothing
        )
                if isnothing(outside_reference)
                        mins = map(minimum, eachrow(stl.points))
                        maxs = map(maximum, eachrow(stl.points))

                        w = maxs .- mins
                        outside_reference = mins .- 0.1 .* w
                end

                TriReference(
                        smart_stltree(stl),
                        outside_reference,
                        false
                )
        end

        """
        $TYPEDSIGNATURES

        Obtain Projection from TriReference
        """
        function Projection(
                reference::TriReference,
                x::AbstractVector{Float64}
        )
                tree = reference.triangulation
                interior = reference.interior

                if point_in_polygon(tree, x; outside_reference = reference.reference_point)
                    interior = (interior ? false : true)
                end

                p, _ = tree(x)
                Projection(p, interior)
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

        Simplify triangulation references.

        Reduces triangulation trees to planar signed distance functions
        once a cell is distant from the surface by more than 
        `d > approximation_ratio * norm(c.widths) / 2`.
        """
        function simplify_triref(
            p::Projection, d::Float64, ref::TriReference, c::Cell;
            approximation_ratio::Float64 = 2.0, 
            filter_tris::Bool = false,
        )
            if !(ref.triangulation isa PlaneSDF) # already simplified?
                center = c.origin .+ c.widths ./ 2
                circumradius = norm(c.widths) / 2

                if d > approximation_ratio * circumradius
                    return TriReference(
                        PlaneSDF(p, center),
                        center, p.interior
                    )
                else
                    if filter_tris
                        local_stl = local_stereolitography(ref.triangulation.stl, c)

                        return TriReference(
                            smart_stltree(local_stl), center, p.interior
                        )
                    elseif norm(p.projection .- center) > circumradius * 0.001
                        return TriReference( # re-define reference for shorter ray-tracing
                            ref.triangulation, center, p.interior
                        )
                    end
                end
            end

            ref
        end

        """
        $TYPEDSIGNATURES

        Refine a cell until all criteria are met.
        Called recursively
        """
        function fixed_mesh_refine(
                c::Cell, 
                surfaces::Vector{TriReference},
                local_lengths::Vector{Float64},
                refinement_regions, refinement_lengths,
                growth_ratio::Float64,
                max_length::Float64,
                ghost_layer_ratio::Float64,
                depth::Int64,
                approximation_ratio::Float64,
                filter_triangles_every::Int64,
                _mgrid_depth::Int64,
        )
                center = c.origin .+ c.widths ./ 2

                ds = map(
                         p -> norm(p.projection .- center) * (1 - 2 * p.interior), c.projections
                )

                surfaces = map(
                    (p, ref, d) -> simplify_triref(
                        p, d, ref, c;
                        approximation_ratio = approximation_ratio,
                        filter_tris = filter_triangles_every != 0 && (
                            depth % filter_triangles_every == 0
                        ),
                    ), c.projections, surfaces, ds
                )

                L = maximum(c.widths)
                Lmax = minimum((@. max(abs(ds) * (growth_ratio - 1.0), local_lengths * (2 ^ _mgrid_depth))))

                if length(refinement_regions) > 0
                    ds_refinement = map(
                        rr -> rr(center), refinement_regions
                    )

                    Lmax = min(
                        Lmax,
                        minimum((@. max(ds_refinement * (growth_ratio - 1.0), refinement_lengths * (2 ^ _mgrid_depth)))),
                    )
                end

                Lmax = min(
                    max_length,
                    Lmax
                )

                circumradius = norm(c.widths) / 2

                if L <= Lmax
                    if minimum(ds) < ghost_layer_ratio * circumradius
                        return Cell[]
                    end

                    return [c]
                end

                if minimum(ds) < (ghost_layer_ratio - 1) * circumradius
                    return Cell[]
                end

                new_widths = 0.5 .* c.widths
                map(
                    mult -> let new_origin = c.origin .+ new_widths .* mult
                        new_center = new_origin .+ new_widths ./ 2
                        ch = Cell(
                                new_origin, new_widths,
                                map(ref -> Projection(ref, new_center), surfaces)
                        )

                        ( # return arguments for next iteration
                                ch, 
                                surfaces,
                                local_lengths,
                                refinement_regions, refinement_lengths,
                                growth_ratio,
                                max_length,
                                ghost_layer_ratio,
                                depth + 1,
                                approximation_ratio,
                                filter_triangles_every,
                                _mgrid_depth,
                        )
                    end,
                    Iterators.product(
                                          fill((0, 1), length(center))...
                    )
                ) |> vec
        end

        """
        $TYPEDFIELDS

        Struct to define a mesh
        """
        struct Mesh
            origins::Matrix{Float64}
            widths::Matrix{Float64}
            centers::Matrix{Float64}
            boundary_projections::Dict{String, Matrix{Float64}}
            boundary_in_domain::Dict{String, Vector{Bool}}
            stereolitographies::Dict{String, Stereolitography}
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
        * A maximum cell size (optional);
        * A ratio between the cell circumradius and the SDF threshold past which
            cells are considered to be out of the domain. `ghost_layer_ratio = -2.0` 
            guarantees that a layer of at least two ghost cell layers are included 
            within each solid;
        * A point reference within the domain. If absent, external flow is assumed;
        * An approximation ratio between wall distance and cell circumradius past which
            distance functions are approximated;
        * A number of recursive refinement levels past which the triangles in the provided
            triangulations are filtered to lighter, local topologies.

        Farfield boundaries may be defined with the following syntax:

        ```
        farfield_boundaries = [
            "inlet" => [
                (1, false), # fwd face, first dimension (x)
                (2, false), # left face, second dimension (y)
                (2, true), # right face, second dimension (y)
                (3, false), # bottom face, third dimension (z)
                (3, true), # top face, third dimension (z)
            ],
            "outlet" => [(1, true)]
        ]
        ```
        """
        function FixedMesh(
                origin::Vector{Float64}, widths::Vector{Float64},
                surfaces::Tuple{String, Stereolitography, Float64}...;
                refinement_regions::AbstractVector = [],
                growth_ratio::Float64 = 1.2,
                max_length::Float64 = Inf,
                ghost_layer_ratio::Float64 = -2.2,
                interior_point = nothing,
                approximation_ratio::Float64 = 2.0,
                filter_triangles_every::Int64 = 0,
                verbose::Bool = false,
                farfield_boundaries = nothing,
                _mgrid_depth::Int64 = 0
        )
            bnames = map(p -> p[1], surfaces) |> collect
            bdries = map(p -> p[2], surfaces) |> collect
            bdry_lengths = map(p -> p[3], surfaces) |> collect

            all_bnames = copy(bnames)
            if !isnothing(farfield_boundaries)
                for (bname, _) in farfield_boundaries
                    push!(all_bnames, bname)
                end
            end

            if length(all_bnames) != length(unique(all_bnames))
                throw(error("Non-unique boundary names in mesh generation"))
            end

            refs = map(p -> p[1], refinement_regions)
            ref_lengths = map(p -> p[2], refinement_regions)

            trirefs = map(
                bdry -> TriReference(bdry; outside_reference = interior_point),
                bdries
            )

            center = origin .+ widths ./ 2
            projs = map(
                ref -> Projection(ref, center), trirefs
            )

            args = [
                (
                    Cell(origin, widths, projs),
                    trirefs, bdry_lengths,
                    refs, ref_lengths,
                                    growth_ratio,
                                    max_length,
                                    ghost_layer_ratio,
                                    1,
                                    approximation_ratio,
                                    filter_triangles_every,
                                    _mgrid_depth,
                )
            ]

            if verbose
                println("===================")
                println("Starting iteration for $(length(origin))-D fixed mesh")
            end

            k = 0
            while !all(a -> isa(a, Cell), args)
                k += 1
                verbose && print("Iteration $k: ")

                args = let rets = Vector{Any}(undef, length(args))
                    @threads for i = 1:length(args)
                        if isa(args[i], Cell)
                            rets[i] = args[i]
                        else
                            rets[i] = fixed_mesh_refine(args[i]...)
                        end
                    end
                    rets
                end |> x -> reduce(vcat, x)

                verbose && println("$(length(args)) cells")
            end
            verbose && println("Iteration ended!!")
            verbose && println("===================")

            bdries_proj = Dict(
                [
                    bname => map(
                        c -> c.projections[i].projection, args
                    ) |> x -> reduce(hcat, x) for (i, bname) in enumerate(bnames)
                ]...
            )
            bdries_in_domain = Dict(
                [
                    bname => map(
                        c -> !(c.projections[i].interior), args
                    ) for (i, bname) in enumerate(bnames)
                ]...
            )

            # define projections to hypercube boundaries
            centers = map(c -> c.origin .+ c.widths ./ 2, args) |> x -> reduce(hcat, x)
            if !isnothing(farfield_boundaries)
                for (bname, tups) in farfield_boundaries
                    projs = similar(centers)
                    dists = fill(Inf64, size(centers, 2))

                    in_domain = trues(size(centers, 2))

                    for (dim, front) in tups
                        for (i, (c, p)) in zip(eachcol(centers), eachcol(projs)) |> enumerate
                            _p = copy(c)
                            if front
                                _p[dim] = origin[dim] + widths[dim]
                            else
                                _p[dim] = origin[dim]
                            end

                            d = norm(_p .- c)

                            if d < dists[i]
                                dists[i] = d
                                p .= _p
                            end
                        end
                    end

                    bdries_proj[bname] = projs
                    bdries_in_domain[bname] = in_domain
                end
            end

            Mesh(
                map(c -> c.origin, args) |> x -> reduce(hcat, x),
                map(c -> c.widths, args) |> x -> reduce(hcat, x),
                centers,
                bdries_proj, bdries_in_domain,
                Dict(
                     [bn => deepcopy(bd) for (bn, bd) in zip(bnames, bdries)]...
                )
            )
        end

        """
        $TYPEDSIGNATURES

        Obtain multiple meshes for a multigrid approach, returned from finest to coarsest.
        The element size ratio between meshes is 2.

        All arguments and kwargs are passed to `FixedMesh`.
        """
        Multigrid(ngrids::Int64, args...; 
            verbose::Bool = false, kwargs...) = map(
            level -> begin
                if verbose
                    println("====Mesh level $level====")
                end

                FixedMesh(args...; kwargs..., 
                    _mgrid_depth = level - 1, verbose = verbose)
            end,
            1:ngrids
        )

        """
        $TYPEDSIGNATURES

        Write cells to VTK file. Kwargs are written as cell data
        """
        function WriteVTK.vtk_grid(
            fname::String, msh::Mesh; kwargs...
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
                grid[String(k)] = v
            end

            grid
        end

        """
        $TYPEDSIGNATURES

        Get number of cells in mesh
        """
        Base.length(msh::Mesh) = size(msh.centers, 2)

        """
        $TYPEDSIGNATURES

        Save mesh to .json format
        """
        mesh2json(fname::String, msh::Mesh) = let fobj = open(fname, "w")
            let nt = (
                origins = msh.origins,
                widths = msh.widths,
                centers = msh.centers,
                boundary_projections = msh.boundary_projections,
                boundary_in_domain = Dict(
                    [k => Int64.(v) for (k, v) in msh.boundary_in_domain]...
                ),
                stereolitographies = Dict(
                                          [k => Dict(
                                                     ["points" => v.points, "simplices" =>v.simplices]...
                                          ) for (k, v) in msh.stereolitographies]...
                )
            )
                json = JSON.json(nt)

                print(fobj, json)
            end
        end

        """
        $TYPEDSIGNATURES

        Recover mesh from .json format
        """
        json2mesh(fname::String) = let d = JSON.parsefile(fname)
            Mesh(
                d["origins"] |> x -> reduce(hcat, x),
                d["widths"] |> x -> reduce(hcat, x),
                d["centers"] |> x -> reduce(hcat, x),
                Dict(
                    [k => reduce(hcat, v) for (k, v) in d["boundary_projections"]]...
                ), Dict(
                    [k => Bool.(v) for (k, v) in d["boundary_in_domain"]]...
                ), Dict(
                    [k => Stereolitography(
                                           reduce(hcat, v["points"]) |> x -> Float64.(x), 
                                           reduce(hcat, v["simplices"]) |> x -> Int64.(x)
                           ) for (k, v) in d["stereolitographies"]]...
                )
            )
        end

end
