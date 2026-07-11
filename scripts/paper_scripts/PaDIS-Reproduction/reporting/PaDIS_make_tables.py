"""Generate publication-ready PaDIS tables from the verification CSV."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reconstruction"))

from LION.utils.paths import LION_EXPERIMENTS_PATH
from PaDIS_identifiers import canonical_experiment, canonical_method

DEFAULT_RECONSTRUCTION_ROOT = (
    LION_EXPERIMENTS_PATH
    / "PaDIS"
    / "final_real_runs"
    / "PaDIS-Reproduction-GCP_reconstruction"
)
DEFAULT_INPUT_CSV = (
    DEFAULT_RECONSTRUCTION_ROOT / "reconstruction_matrix_verification.csv"
)
DEFAULT_OUTPUT_DIR = LION_EXPERIMENTS_PATH / "PaDIS" / "paper_tables"
DEFAULT_OUTPUT_TEX = DEFAULT_OUTPUT_DIR / "reconstruction_tables.tex"
DEFAULT_CSV_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "csv"
DEFAULT_GCP_LOG_ROOT = (
    DEFAULT_RECONSTRUCTION_ROOT / ".manual_gcp_reconstruction" / "logs"
)

IMPLEMENTATION = {
    "paper": "Paper",
    "public_repo": "Public-compatible",
    "lion_physics": "LION-physics",
}
METHOD_ORDER = {
    name: index
    for index, name in enumerate(
        (
            "baseline",
            "cp_tv",
            "pnp_admm",
            "whole_image_diffusion",
            "langevin",
            "predictor_corrector",
            "ve_ddnm",
            "patch_average",
            "patch_stitch",
            "padis_dps",
        )
    )
}
IMPLEMENTATION_ORDER = {"paper": 0, "public_repo": 1, "lion_physics": 2}
METRICS = (
    ("psnr", "PSNR", 2),
    ("ssim", "SSIM", 3),
    ("mae", "MAE", 4),
    ("relative_sinogram_residual", "RSR", 4),
)


def _method_labels(row: dict[str, str]) -> tuple[str, str]:
    """Handle method labels for the PaDIS workflow."""
    method = row["method"]
    prior = "Whole-Image" if row["prior_mode"] == "whole_image" else "PaDIS"
    if method == "baseline":
        return "FDK", "--"
    if method == "cp_tv":
        return "CP", "TV"
    if method == "pnp_admm":
        conditioned = row["matrix_group"] == "pnp_noise_conditioned"
        return "PnP-ADMM", "DRUnet (cond)" if conditioned else "DRUnet"
    if method in {"whole_image_diffusion", "padis_dps"}:
        return "VE-DPS", prior
    if method == "langevin":
        return "Langevin", prior
    if method == "predictor_corrector":
        return "PC", prior
    if method == "ve_ddnm":
        return "VE-DDNM", prior
    if method == "patch_average":
        return "VE-DPS", "Patch avg."
    if method == "patch_stitch":
        return "VE-DPS", "Patch stitch."
    raise ValueError(f"Unknown method: {method}")


def _display_metric(row: dict[str, str], stem: str, decimals: int) -> str:
    """Return display-ready metric."""
    mean = float(row[f"mean_{stem}"])
    low = float(row[f"mean_{stem}_bootstrap_ci_low"])
    high = float(row[f"mean_{stem}_bootstrap_ci_high"])
    return (
        f"{mean:.{decimals}f} (+{high - mean:.{decimals}f}/-{mean - low:.{decimals}f})"
    )


def _metric_cells(row: dict[str, str] | None, prefix: str = "") -> dict[str, str]:
    """Return metric cells."""
    if row is None:
        return {f"{prefix}{label}": "--" for _stem, label, _decimals in METRICS}
    return {
        f"{prefix}{label}": _display_metric(row, stem, decimals)
        for stem, label, decimals in METRICS
    }


def _write_csv(
    path: Path, records: list[dict[str, str]], *, allow_missing: bool = False
) -> bool:
    """Write csv."""
    if not records:
        if allow_missing:
            return False
        raise ValueError(f"No records generated for {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return True


def calculate_timing_rows(
    mode: str, log_root: Path, jobs_json: Path
) -> list[dict[str, str]]:
    """Calculate mean seconds per reconstructed slice from runner progress logs."""
    jobs = json.loads(jobs_json.read_text(encoding="utf-8"))
    pattern = "reconstruction_*.log" if mode in {"gcp", "colab"} else "slurm-*.out"
    grouped: dict[tuple[str, str], list[float]] = {}
    labels = {
        "baseline": "FDK",
        "cp_tv": "CP",
        "pnp_admm": "PnP-ADMM",
        "whole_image_diffusion": "VE-DPS (Whole-Image)",
        "langevin": "Langevin",
        "predictor_corrector": "Predictor-corrector",
        "ve_ddnm": "VE-DDNM",
        "patch_average": "Patch average",
        "patch_stitch": "Patch stitch",
        "padis_dps": "PaDIS-DPS",
    }
    for path in log_root.rglob(pattern):
        index_match = re.search(
            r"reconstruction_(\d+)" if mode in {"gcp", "colab"} else r"_(\d+)\.out$",
            path.name,
        )
        elapsed_matches = re.findall(
            r"LIDC test run:\s*100%\|[^\r\n]*?\d+/\d+ \[[^\r\n]*?,\s*([0-9.]+)s/it\]",
            path.read_text(encoding="utf-8", errors="ignore"),
        )
        if not index_match or not elapsed_matches:
            continue
        index = int(index_match.group(1))
        if index >= len(jobs):
            continue
        job = jobs[index]
        if (
            job.get("experiment") != "ct_20"
            or job.get("matrix_group", "main") != "main"
        ):
            continue
        grouped.setdefault((job["implementation"], job["method"]), []).append(
            float(elapsed_matches[-1])
        )
    if not grouped:
        raise ValueError(f"No completed ct_20 timing records found below {log_root}")
    return [
        {
            "Implementation": IMPLEMENTATION[implementation],
            "Reconstruction method": labels[method],
            "Mean time per slice": f"{sum(values) / len(values):.2f} s",
        }
        for (implementation, method), values in sorted(
            grouped.items(),
            key=lambda item: (
                IMPLEMENTATION_ORDER[item[0][0]],
                METHOD_ORDER[item[0][1]],
            ),
        )
    ]


def export_table_csvs(
    rows: list[dict[str, str]],
    output_dir: str | Path,
    *,
    timing_rows=None,
    allow_missing: bool = False,
) -> list[Path]:
    """Write one decoded, human-readable CSV for each generated table."""
    rows = [
        {
            **row,
            "method": canonical_method(row["method"]),
            "experiment": canonical_experiment(row["experiment"]),
        }
        for row in rows
    ]
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def paired(experiments: tuple[str, ...], methods: set[str], partial: bool = False):
        """Provide the paired callback used by the enclosing operation."""
        grouped: dict[tuple[str, str, str], dict[str, dict[str, str]]] = {}
        for row in rows:
            if row["matrix_group"] not in {"main", "pnp_noise_conditioned"}:
                continue
            if row["experiment"] not in experiments or row["method"] not in methods:
                continue
            variant = (
                "conditioned"
                if row["matrix_group"] == "pnp_noise_conditioned"
                else row["prior_mode"]
            )
            grouped.setdefault((row["implementation"], row["method"], variant), {})[
                row["experiment"]
            ] = row
        items = [
            (key, values)
            for key, values in grouped.items()
            if partial or all(experiment in values for experiment in experiments)
        ]
        return sorted(
            items,
            key=lambda item: (
                IMPLEMENTATION_ORDER[item[0][0]],
                METHOD_ORDER[item[0][1]],
                item[0][2],
            ),
        )

    main_methods = set(METHOD_ORDER)
    table1: list[dict[str, str]] = []
    for (implementation, _method, _variant), values in paired(
        ("ct_20", "ct_8"), main_methods, True
    ):
        source = values.get("ct_20") or values["ct_8"]
        sampler, prior = _method_labels(source)
        record = {
            "Implementation": IMPLEMENTATION[implementation],
            "Sampler": sampler,
            "Prior": prior,
        }
        record.update(_metric_cells(values.get("ct_20"), "20-View 360 Degrees "))
        record.update(_metric_cells(values.get("ct_8"), "8-View 360 Degrees "))
        table1.append(record)
    path = output_dir / "table_1_ct_reconstruction.csv"
    if _write_csv(path, table1, allow_missing=allow_missing):
        written.append(path)

    extra_methods = {
        "baseline",
        "cp_tv",
        "whole_image_diffusion",
        "ve_ddnm",
        "padis_dps",
    }
    table2: list[dict[str, str]] = []
    for (implementation, _method, _variant), values in paired(
        ("ct_60", "ct_20_limited_angle_120"), extra_methods
    ):
        source = values["ct_60"]
        sampler, prior = _method_labels(source)
        record = {
            "Implementation": IMPLEMENTATION[implementation],
            "Sampler": sampler,
            "Prior": prior,
        }
        record.update(_metric_cells(values["ct_60"], "60-View 360 Degrees "))
        record.update(
            _metric_cells(values["ct_20_limited_angle_120"], "20-View 120 Degrees ")
        )
        table2.append(record)
    path = output_dir / "table_2_additional_geometries.csv"
    if _write_csv(path, table2, allow_missing=allow_missing):
        written.append(path)

    table3: list[dict[str, str]] = []
    selected = [
        row
        for row in rows
        if row["matrix_group"] == "main"
        and row["experiment"] == "ct_512_60"
        and row["method"] in {"baseline", "cp_tv", "padis_dps"}
    ]
    for row in sorted(
        selected,
        key=lambda r: (
            IMPLEMENTATION_ORDER[r["implementation"]],
            METHOD_ORDER[r["method"]],
        ),
    ):
        sampler, prior = _method_labels(row)
        record = {
            "Implementation": IMPLEMENTATION[row["implementation"]],
            "Sampler": sampler,
            "Prior": prior,
        }
        record.update(_metric_cells(row))
        table3.append(record)
    path = output_dir / "table_3_512_reconstruction.csv"
    if _write_csv(path, table3, allow_missing=allow_missing):
        written.append(path)

    table4_rows = [
        row
        for row in rows
        if row["experiment"] == "ct_20"
        and (
            row["matrix_group"].startswith("patch_size_p")
            or (
                row["matrix_group"] == "main"
                and row["method"] == "whole_image_diffusion"
            )
        )
    ]
    table4: list[dict[str, str]] = []

    def patch_size(row: dict[str, str]) -> str:
        """Handle patch size for the PaDIS workflow."""
        return (
            "P=256 (Whole-Image)"
            if row["method"] == "whole_image_diffusion"
            else f"P={row['matrix_group'].split('p')[-1]}"
        )

    for row in sorted(
        table4_rows,
        key=lambda r: (
            IMPLEMENTATION_ORDER[r["implementation"]],
            int(patch_size(r).split("=")[1].split()[0]),
        ),
    ):
        record = {
            "Implementation": IMPLEMENTATION[row["implementation"]],
            "Patch size": patch_size(row),
        }
        record.update(_metric_cells(row))
        table4.append(record)
    path = output_dir / "table_4_patch_size_ablation.csv"
    if _write_csv(path, table4, allow_missing=allow_missing):
        written.append(path)

    table5_rows = [
        row
        for row in rows
        if row["experiment"] == "ct_20"
        and row["matrix_group"].startswith("dataset_size_")
    ]
    table5: list[dict[str, str]] = []
    for row in sorted(
        table5_rows,
        key=lambda r: (IMPLEMENTATION_ORDER[r["implementation"]], r["matrix_group"]),
    ):
        record = {
            "Implementation": IMPLEMENTATION[row["implementation"]],
            "Dataset": "Full" if row["matrix_group"].endswith("_full") else "Default",
            "Prior": "Whole-Image" if row["prior_mode"] == "whole_image" else "PaDIS",
        }
        record.update(_metric_cells(row))
        table5.append(record)
    path = output_dir / "table_5_dataset_size_ablation.csv"
    if _write_csv(path, table5, allow_missing=allow_missing):
        written.append(path)

    table6_rows = [
        row
        for row in rows
        if row["experiment"] == "ct_20"
        and (
            row["matrix_group"].startswith("position_no_encoding_")
            or row["matrix_group"].startswith("schedule_")
        )
    ]
    table6: list[dict[str, str]] = []

    def settings(row: dict[str, str]) -> tuple[str, str, str]:
        """Provide the settings callback used by the enclosing operation."""
        group = row["matrix_group"]
        position = "No" if group.startswith("position_no_encoding_") else "Yes"
        initialization = "FDK" if group.endswith("fdk_init") else "Noise"
        schedule = "EDM" if group.startswith("schedule_edm_") else "Geometric"
        return position, initialization, schedule

    for row in sorted(
        table6_rows,
        key=lambda r: (IMPLEMENTATION_ORDER[r["implementation"]], settings(r)),
    ):
        position, initialization, schedule = settings(row)
        record = {
            "Implementation": IMPLEMENTATION[row["implementation"]],
            "Position": position,
            "Initialization": initialization,
            "Schedule": schedule,
        }
        record.update(_metric_cells(row))
        table6.append(record)
    path = output_dir / "table_6_padis_ablation.csv"
    if _write_csv(path, table6, allow_missing=allow_missing):
        written.append(path)

    if not timing_rows:
        raise ValueError("Timing rows must be calculated from runner logs")
    path = output_dir / "table_7_timings.csv"
    if _write_csv(path, timing_rows, allow_missing=allow_missing):
        written.append(path)
    return written


TABLE_METADATA = (
    (
        "table_1_ct_reconstruction.csv",
        "CT reconstruction results.",
        "tab:ct-reconstruction",
    ),
    (
        "table_2_additional_geometries.csv",
        "Additional CT geometry results.",
        "tab:additional-geometries",
    ),
    (
        "table_3_512_reconstruction.csv",
        "Native-resolution CT reconstruction results.",
        "tab:512-reconstruction",
    ),
    (
        "table_4_patch_size_ablation.csv",
        "Patch-size ablation results.",
        "tab:patch-size-ablation",
    ),
    (
        "table_5_dataset_size_ablation.csv",
        "Training-dataset-size ablation results.",
        "tab:dataset-size-ablation",
    ),
    (
        "table_6_padis_ablation.csv",
        "PaDIS sampling and position-encoding ablations.",
        "tab:padis-ablation",
    ),
    (
        "table_7_timings.csv",
        "Mean reconstruction time per slice.",
        "tab:reconstruction-timings",
    ),
)


def _latex_escape(value: str) -> str:
    """Handle latex escape for the PaDIS workflow."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)


def write_latex_tables(csv_paths: list[Path], tex_path: str | Path) -> Path:
    """Write a self-contained LaTeX fragment from the decoded table CSVs."""
    tex_path = Path(tex_path).expanduser().resolve()
    metadata = {name: (caption, label) for name, caption, label in TABLE_METADATA}
    blocks = [
        "% Generated by scripts/paper_scripts/PaDIS-Reproduction/reporting/PaDIS_make_tables.py.",
        "% Requires the booktabs and graphicx packages.",
        "",
    ]
    for csv_path in csv_paths:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"No records found in {csv_path}")
        headers = list(rows[0])
        caption, label = metadata[csv_path.name]
        column_spec = "l" * len(headers)
        blocks.extend(
            [
                r"\begin{table}[ht]",
                r"  \centering",
                f"  \\caption{{{caption}}}",
                f"  \\label{{{label}}}",
                r"  \resizebox{\textwidth}{!}{%",
                f"  \\begin{{tabular}}{{{column_spec}}}",
                r"    \toprule",
                "    " + " & ".join(_latex_escape(value) for value in headers) + r" \\",
                r"    \midrule",
            ]
        )
        blocks.extend(
            "    "
            + " & ".join(_latex_escape(row[header]) for header in headers)
            + r" \\"
            for row in rows
        )
        blocks.extend(
            [
                r"    \bottomrule",
                r"  \end{tabular}%",
                r"  }",
                r"\end{table}",
                "",
            ]
        )
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(blocks), encoding="utf-8")
    return tex_path


def csv_to_latex_tables(
    csv_path: str | Path,
    tex_path: str | Path,
    *,
    generator_path: str | Path | None = None,
    csv_output_dir: str | Path | None = None,
    timing_rows: list[dict[str, str]] | None = None,
    allow_missing: bool = False,
) -> Path:
    """Parse ``csv_path`` and write the complete LaTeX table document.

    The generated document contains the current seven publication-ready tables,
    including asymmetric bootstrap intervals, displayed-value tie bolding,
    implementation blocks, sampler/prior labels, ablations, missing-value marks,
    and the supplied timing table.

    Parameters
    ----------
    csv_path:
        Verification CSV with the columns used by
        ``reconstruction_matrix_verification.csv``.
    tex_path:
        Destination ``.tex`` file.
    generator_path:
        Optional external table-layout generator. When omitted, the integrated
        LION generator writes the LaTeX fragment directly.
    csv_output_dir:
        Directory for the seven decoded per-table CSV files. Defaults to a
        ``table_csvs`` folder beside ``tex_path``.

    Returns
    -------
    pathlib.Path
        The resolved path of the generated LaTeX file.
    """
    csv_path = Path(csv_path).expanduser().resolve()
    tex_path = Path(tex_path).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    tex_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    csv_paths = export_table_csvs(
        rows,
        csv_output_dir or DEFAULT_CSV_OUTPUT_DIR,
        timing_rows=timing_rows,
        allow_missing=allow_missing,
    )

    if generator_path is None:
        write_latex_tables(csv_paths, tex_path)
    else:
        generator = Path(generator_path).expanduser().resolve()
        if not generator.is_file():
            raise FileNotFoundError(f"Table generator not found: {generator}")
        env = os.environ.copy()
        env["RECONSTRUCTION_TABLES_CSV"] = str(csv_path)
        env["RECONSTRUCTION_TABLES_TEX"] = str(tex_path)
        subprocess.run([sys.executable, str(generator)], env=env, check=True)

    if not tex_path.is_file():
        raise RuntimeError(f"Generator did not create: {tex_path}")
    return tex_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the table-generation command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--tex-path", type=Path, default=DEFAULT_OUTPUT_TEX)
    parser.add_argument("--csv-output-dir", type=Path, default=DEFAULT_CSV_OUTPUT_DIR)
    parser.add_argument("--generator", type=Path, default=None)
    parser.add_argument(
        "--timing-mode", choices=("gcp", "colab", "slurm"), default="gcp"
    )
    parser.add_argument("--timing-log-root", type=Path, default=None)
    parser.add_argument("--timing-jobs-json", type=Path, default=None)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Generate only tables represented in a partial smoke-run CSV.",
    )
    return parser


def main() -> None:
    """Generate decoded CSV and LaTeX tables from verified metrics."""
    args = build_arg_parser().parse_args()
    log_root = args.timing_log_root
    if log_root is None:
        log_root = (
            DEFAULT_GCP_LOG_ROOT if args.timing_mode in {"gcp", "colab"} else Path.cwd()
        )
    jobs_json = (
        args.timing_jobs_json
        or DEFAULT_RECONSTRUCTION_ROOT / "reconstruction_matrix_jobs.json"
    )
    try:
        timing_rows = calculate_timing_rows(args.timing_mode, log_root, jobs_json)
    except ValueError:
        if not args.allow_missing:
            raise
        timing_rows = []
    output = csv_to_latex_tables(
        args.csv_path,
        args.tex_path,
        generator_path=args.generator,
        csv_output_dir=args.csv_output_dir,
        timing_rows=timing_rows,
        allow_missing=args.allow_missing,
    )
    print(f"Saved PaDIS tables to {output}")


if __name__ == "__main__":
    main()
