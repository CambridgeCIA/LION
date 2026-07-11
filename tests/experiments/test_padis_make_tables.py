import csv

from scripts.paper_scripts.PaDIS.PaDIS_make_tables import (
    DEFAULT_CSV_OUTPUT_DIR,
    DEFAULT_INPUT_CSV,
    DEFAULT_OUTPUT_TEX,
    _method_labels,
    build_arg_parser,
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
