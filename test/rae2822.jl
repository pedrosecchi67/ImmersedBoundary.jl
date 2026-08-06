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

    coarse_doms, coarseners, prolongators = multigrid(dom)
    
    export_vtk("rae2822", dom)
    export_vtk("rae2822_coarse", coarse_doms[end])
end
