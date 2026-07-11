import csv
import json

from PaDIS_make_tables import (
    DEFAULT_CSV_OUTPUT_DIR,
    DEFAULT_INPUT_CSV,
    DEFAULT_OUTPUT_TEX,
    _method_labels,
    build_arg_parser,
    calculate_timing_rows,
    write_latex_tables,
)


def test_table_cli_uses_standard_padis_paths():
    args = build_arg_parser().parse_args([])

    assert args.csv_path == DEFAULT_INPUT_CSV
    assert args.tex_path == DEFAULT_OUTPUT_TEX
    assert args.csv_output_dir == DEFAULT_CSV_OUTPUT_DIR
    assert args.generator is None


def test_fanbeam_analytic_baseline_is_labelled_fdk():
    assert _method_labels(
        {
            "method": "baseline",
            "prior_mode": "patch",
            "matrix_group": "main",
        }
    ) == ("FDK", "--")


def test_historical_admm_tv_identifier_is_presented_as_cp_with_tv_prior():
    assert _method_labels(
        {
            "method": "cp_tv",
            "prior_mode": "patch",
            "matrix_group": "main",
        }
    ) == ("CP", "TV")


def test_integrated_latex_writer_creates_table_fragment(tmp_path):
    csv_path = tmp_path / "table_7_timings.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("Implementation", "Reconstruction method"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "Implementation": "LION-physics",
                "Reconstruction method": "FDK & baseline",
            }
        )

    output = write_latex_tables([csv_path], tmp_path / "tables.tex")
    text = output.read_text(encoding="utf-8")

    assert "\\begin{table}" in text
    assert "LION-physics" in text
    assert r"FDK \& baseline" in text
    assert "tab:reconstruction-timings" in text


def test_timing_calculation_supports_gcp_and_slurm_log_names(tmp_path):
    jobs = [
        {
            "implementation": "lion_physics",
            "method": "cp_tv",
            "experiment": "ct_20",
            "matrix_group": "main",
        }
    ]
    manifest = tmp_path / "jobs.json"
    manifest.write_text(json.dumps(jobs), encoding="utf-8")
    progress = "LIDC test run: 100%|##########| 25/25 [01:00<00:00, 2.50s/it]\n"
    gcp = tmp_path / "gcp"
    gcp.mkdir()
    (gcp / "reconstruction_000000.reconstruction.gpu0.slot1.log").write_text(progress)
    slurm = tmp_path / "slurm"
    slurm.mkdir()
    (slurm / "slurm-PaDIS_recon-123_0.out").write_text(progress)

    for mode, root in (("gcp", gcp), ("colab", gcp), ("slurm", slurm)):
        assert calculate_timing_rows(mode, root, manifest) == [
            {
                "Implementation": "LION-physics",
                "Reconstruction method": "CP",
                "Mean time per slice": "2.50 s",
            }
        ]
