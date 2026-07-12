"""Test padis paper figures behaviour."""

import torch

from PaDIS_make_paper_figures import (
    ADDITIONAL_EXAMPLE_OFFSETS,
    IMPLEMENTED_FIGURES,
    PATCH_SIZE_EXAMPLE_OFFSETS,
    TWO_EXAMPLE_OFFSETS,
    body_hu_percentile_range,
    build_arg_parser,
    figure_specs,
    recon_path,
    selected_figures,
    should_show_panel_title,
    target_bbox,
)


def test_recon_path_falls_back_to_legacy_identifiers(tmp_path):
    """Verify that recon path falls back to legacy identifiers."""
    legacy = (
        tmp_path
        / "admm_tv/patch_lidc_default/lion_physics/lion/ct_fanbeam_180"
        / "reconstructions.pt"
    )
    legacy.parent.mkdir(parents=True)
    legacy.touch()

    assert (
        recon_path(
            tmp_path,
            method="cp_tv",
            model="patch_lidc_default",
            implementation="lion_physics",
            experiment="ct_20_limited_angle_120",
        )
        == legacy
    )


def test_paper_figure_builder_lists_requested_implemented_figures(tmp_path):
    """Verify that paper figure builder lists requested implemented figures."""
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
    assert [row[0].row for row in specs["figure4_generation"].panels] == [
        "Whole image",
        "Patch stitching",
        "Patch averaging",
        "PaDIS",
    ]
    assert "paper-generation-naive-patch" not in {
        panel.path.parent.name
        for row in specs["figure4_generation"].panels
        for panel in row
    }


def test_paper_figure_specs_follow_multi_sample_ct_layouts(tmp_path):
    """Verify that paper figure specs follow multi sample ct layouts."""
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
    assert all(row[0].title == "FDK" for row in figure5.panels)
    assert all(row[1].title == "CP" for row in figure5.panels)
    assert not any(
        panel.title == "ADMM-TV"
        for spec in specs.values()
        for row in spec.panels
        for panel in row
    )
    assert [row[0].row for row in figure5.panels] == [
        "60 views\n360° range\nSample 1",
        "60 views\n360° range\nSample 6",
        "20 views\n360° range\nSample 1",
        "20 views\n360° range\nSample 6",
    ]
    assert [[panel.sample_index for panel in row] for row in figure5.panels] == [
        [3, 3, 3, 3, 3],
        [8, 8, 8, 8, 8],
        [3, 3, 3, 3, 3],
        [8, 8, 8, 8, 8],
    ]
    assert {panel.window for panel in figure5.panels[0]} == {"soft_tissue"}
    assert {panel.window for panel in figure5.panels[2]} == {"normal"}
    assert not any(
        "Slice" in panel.row
        for spec in specs.values()
        for row in spec.panels
        for panel in row
    )

    for key, experiment in (
        ("figureA1_ct20_additional", "ct_20"),
        ("figureA2_ct8_additional", "ct_8"),
    ):
        spec = specs[key]
        assert len(spec.panels) == 7
        assert all(len(row) == 5 for row in spec.panels)
        assert [row[0].sample_index for row in spec.panels] == [
            3 + offset for offset in ADDITIONAL_EXAMPLE_OFFSETS
        ]
        assert all("360° range" in row[0].row for row in spec.panels)
        assert {
            panel.path
            for row in spec.panels
            for panel in row
            if panel.source == "reconstruction"
        } <= {
            tmp_path
            / f"recon/baseline/patch_lidc_default/lion_physics/lion/{experiment}/reconstructions.pt",
            tmp_path
            / f"recon/cp_tv/patch_lidc_default/lion_physics/lion/{experiment}/reconstructions.pt",
            tmp_path
            / f"recon/whole_image_diffusion/whole_lidc_default/lion_physics/lion/{experiment}/reconstructions.pt",
            tmp_path
            / f"recon/padis_dps/patch_lidc_default/lion_physics/lion/{experiment}/reconstructions.pt",
        }


def test_patch_size_figure_is_vertical_and_uses_four_samples(tmp_path):
    """Verify that patch size figure is vertical and uses four samples."""
    specs = {
        spec.name: spec
        for spec in figure_specs(
            tmp_path / "recon",
            tmp_path / "gen",
            sample_index=2,
        )
    }

    assert PATCH_SIZE_EXAMPLE_OFFSETS == (0, 1, 5, 6)
    assert len(specs["figureA5_patch_size"].panels) == 6
    assert [
        [panel.sample_index for panel in row]
        for row in specs["figureA5_patch_size"].panels
    ] == [[2, 3, 7, 8]] * 6
    assert [row[0].row for row in specs["figureA5_patch_size"].panels] == [
        "8 × 8 patches",
        "16 × 16 patches",
        "32 × 32 patches",
        "56 × 56 patches",
        "96 × 96 patches",
        "Ground truth",
    ]


def test_position_figure_uses_two_example_samples(tmp_path):
    """Verify that position figure uses two example samples."""
    specs = {
        spec.name: spec
        for spec in figure_specs(
            tmp_path / "recon",
            tmp_path / "gen",
            sample_index=2,
        )
    }

    assert len(specs["figureA7_position_encoding"].panels) == 2
    assert [
        [panel.sample_index for panel in row]
        for row in specs["figureA7_position_encoding"].panels
    ] == [[2, 2, 2, 2, 2], [7, 7, 7, 7, 7]]


def test_representative_and_additional_examples_are_disjoint():
    """Verify that representative and additional examples are disjoint."""
    assert TWO_EXAMPLE_OFFSETS == (0, 5)
    assert len(ADDITIONAL_EXAMPLE_OFFSETS) == 7
    assert set(TWO_EXAMPLE_OFFSETS).isdisjoint(ADDITIONAL_EXAMPLE_OFFSETS)


def test_repeated_column_headings_are_only_shown_once(tmp_path):
    """Verify that repeated column headings are only shown once."""
    specs = {
        spec.name: spec for spec in figure_specs(tmp_path / "recon", tmp_path / "gen")
    }
    figure5 = specs["figure5_ct_reconstruction"]

    assert all(
        should_show_panel_title(figure5.panels, 0, col_index)
        for col_index in range(len(figure5.panels[0]))
    )
    assert not any(
        should_show_panel_title(figure5.panels, 1, col_index)
        for col_index in range(len(figure5.panels[1]))
    )


def test_body_crop_is_square_for_consistent_row_layout(tmp_path):
    """Verify that body crop is square for consistent row layout."""
    path = tmp_path / "reconstructions.pt"
    target = torch.zeros(1, 1, 64, 64)
    target[..., 10:42, 20:50] = 1.0
    torch.save({"targets": target}, path)

    top, bottom, left, right = target_bbox(path, 0, pad=3)

    assert bottom - top == right - left
    assert top == 64 - bottom
    assert left == 64 - right
    assert top <= 10 and bottom >= 42
    assert left <= 20 and right >= 50


def test_body_crop_default_has_no_extra_padding():
    """Verify that body crop default has no extra padding."""
    parser = build_arg_parser()

    assert (
        parser.parse_args(
            [
                "--reconstruction-root",
                "/tmp/recon",
                "--output-folder",
                "/tmp/figures",
            ]
        ).body_bbox_padding
        == 0
    )


def test_hu_display_range_uses_body_15th_and_95th_percentiles(tmp_path):
    """Verify that hu display range uses body 15th and 95th percentiles."""
    path = tmp_path / "reconstructions.pt"
    hu_values = torch.tensor((-800.0, -400.0, 0.0, 400.0, 800.0))
    normalised_values = (hu_values + 1000.0) / 3000.0
    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, 1, 1:4] = normalised_values[:3]
    target[0, 0, 2, 1:3] = normalised_values[3:]
    torch.save({"targets": target}, path)

    lower, upper = body_hu_percentile_range(path, 0)

    assert abs(lower - (-560.0)) < 1e-4
    assert abs(upper - 720.0) < 1e-4
