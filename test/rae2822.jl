begin
    @info "Running RAE2822 test case..."

    stl = Stereolitography("rae2822.dat") |> merge_points
    features = feature_regions(stl; radius = 0.05) |> DistanceField

    msh = Mesh(
        [-25.0f0, -25.0f0], [50.0f0, 50.0f0],
        ("wall", stl, 1f-2);
        refinement_regions = [
            features => 5f-3,
        ],
        verbose = true,
    )

    dom = Domain(
        msh;
        hypercube_families = [
            "farfield" => [(1, false), (1, true), (2, false), (2, true)],
        ],
        verbose = true,
    )

    X = zeros(Float32, length(dom), ndims(dom))
    dom(X) do part, X
        X .= part.centers;
    end
    CG = volume_integral(dom, X) ./ 2500.0f0
    @show CG

    ny = zeros(Float32, length(dom))
    impose_bc!(dom, "wall", ny) do bdry, ny
        bdry.normals[:, 2]
    end

    coarse_doms, coarseners, prolongators = multigrid(dom)
    
    export_vtk("rae2822", dom; ny = ny,)
    export_vtk("rae2822_coarse", coarse_doms[end])
end
