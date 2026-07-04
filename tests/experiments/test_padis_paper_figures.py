from scripts.paper_scripts.PaDIS.PaDIS_make_paper_figures import (
    IMPLEMENTED_FIGURES,
    figure_specs,
    selected_figures,
)


def test_paper_figure_builder_lists_requested_implemented_figures(tmp_path):
    specs = {
        spec.name: spec for spec in figure_specs(tmp_path / "recon", tmp_path / "gen")
    }

    assert selected_figures("all") == IMPLEMENTED_FIGURES
    assert set(specs) == set(IMPLEMENTED_FIGURES)
    assert "figureA1_ct20_additional" in specs
    assert "figureA2_ct8_additional" in specs
    assert (
        "EDM and DDIM accelerated samplers"
        in specs["figureA9_generation_langevin"].unsupported_note
    )
    assert "heavy deblurring row" in specs["figureA10_extra_ct"].unsupported_note
    assert (
        specs["figure4_generation"].panels[0][0].path
        == tmp_path / "gen/lion-paper-protocol/paper-generation-whole/samples.pt"
    )


def test_paper_figure_specs_follow_multi_slice_ct_layouts(tmp_path):
    specs = {
        spec.name: spec
        for spec in figure_specs(
            tmp_path / "recon",
            tmp_path / "gen",
            sample_index=3,
        )
    }

    figure5 = specs["figure5_ct_reconstruction"]
    assert len(figure5.panels) == 4
    assert [row[0].row for row in figure5.panels] == [
        "60 views",
        "60 views",
        "20 views",
        "20 views",
    ]
    assert [[panel.sample_index for panel in row] for row in figure5.panels] == [
        [3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4],
        [3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4],
    ]
    assert {panel.window for panel in figure5.panels[0]} == {"soft_tissue"}
    assert {panel.window for panel in figure5.panels[2]} == {"normal"}

    for key, experiment in (
        ("figureA1_ct20_additional", "ct_20"),
        ("figureA2_ct8_additional", "ct_8"),
    ):
        spec = specs[key]
        assert len(spec.panels) == 7
        assert all(len(row) == 5 for row in spec.panels)
        assert [row[0].sample_index for row in spec.panels] == list(range(3, 10))
        assert {
            panel.path
            for row in spec.panels
            for panel in row
            if panel.source == "reconstruction"
        } <= {
            tmp_path
            / f"recon/baseline/patch_lidc_default/lion_physics/lion/{experiment}/reconstructions.pt",
            tmp_path
            / f"recon/admm_tv/patch_lidc_default/lion_physics/lion/{experiment}/reconstructions.pt",
            tmp_path
            / f"recon/whole_image_diffusion/whole_lidc_default/lion_physics/lion/{experiment}/reconstructions.pt",
            tmp_path
            / f"recon/padis_dps/patch_lidc_default/lion_physics/lion/{experiment}/reconstructions.pt",
        }


def test_patch_and_position_figures_use_two_example_slices(tmp_path):
    specs = {
        spec.name: spec
        for spec in figure_specs(
            tmp_path / "recon",
            tmp_path / "gen",
            sample_index=2,
        )
    }

    assert len(specs["figureA5_patch_size"].panels) == 2
    assert [
        [panel.sample_index for panel in row]
        for row in specs["figureA5_patch_size"].panels
    ] == [[2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]]

    assert len(specs["figureA7_position_encoding"].panels) == 2
    assert [
        [panel.sample_index for panel in row]
        for row in specs["figureA7_position_encoding"].panels
    ] == [[2, 2, 2, 2, 2], [3, 3, 3, 3, 3]]
