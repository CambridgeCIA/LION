# =============================================================================
# This file is part of LION library
# License : GPL-3
# =============================================================================
"""
Checks of #264 on the real LIDC-IDRI data: the resampling of the processed
slices to one pixel spacing (500 mm over 512x512), the experiments' doses with
it, and the data loader on resampled data. See README.md in this folder for
the order to run them in.

  resampling  CPU, read-only. A few patients across the range of pixel
              spacings, before and after resampling, on the same mm axes.
  doses       GPU, read-only. FDK of a small, a typical and a large patient
              at the experiments' doses: before #264 (each scan's own slices,
              300 mm geometry) and after (resampled slices, 500 mm geometry).
  trial       CPU. Copies the first patients to a separate folder, where the
              pre-processing's --resample-only can be tried without touching
              the real data, and saves the slices it should produce.
  loader      GPU. After --resample-only on the trial copy: checks what it
              wrote against the saved slices, then reconstructs test slices of
              the experiments through LION's data loader.
"""

import argparse
import importlib.util
import json
import pathlib
import shutil
import sqlite3
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "lidc_resampling", REPO / "LION/data_loaders/LIDC_IDRI/resampling.py"
)
resampling = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(resampling)

NEW_SPACING = resampling.PIXEL_SPACING_MM
HU_WINDOW = (-1000, 400)


# --------------------------------------------------------------------------
# Data


def default_data_folder():
    from LION.utils.paths import LIDC_IDRI_PROCESSED_DATASET_PATH

    return pathlib.Path(LIDC_IDRI_PROCESSED_DATASET_PATH)


def original_spacings():
    """{patient id: pixel spacing in mm} from pylidc's database (read with
    sqlite, without importing pylidc), the scan the pre-processing uses."""
    spec = importlib.util.find_spec("pylidc")
    if spec is None:
        raise ImportError("pylidc is needed for the pixel spacings: pip install pylidc")
    database = pathlib.Path(spec.submodule_search_locations[0]) / "pylidc.sqlite"
    rows = sqlite3.connect(database).execute(
        "SELECT patient_id, pixel_spacing FROM scans ORDER BY id"
    )
    spacings = {}
    for patient, spacing in rows:
        spacings.setdefault(patient, spacing)  # pylidc's .first()
    return spacings


def patients(folder):
    return sorted(p.name for p in folder.glob("LIDC-IDRI-????") if p.is_dir())


def is_resampled(folder, patient):
    return (folder / patient / "pixel_spacing.json").is_file()


def require_native(folder, patient):
    if is_resampled(folder, patient):
        raise SystemExit(
            f"{patient} in {folder} is already resampled: these checks compare "
            "with the slices processed before #264. Use --data to point to a "
            "folder that was not converted (or the trial copy before converting)."
        )


def n_slices(folder, patient):
    return len(list((folder / patient).glob("slice_*.npy")))


def masks_of(folder, patient):
    """{slice index: [mask files]}"""
    found = {}
    for path in (folder / patient).glob("mask_*.npy"):
        found.setdefault(int(path.name.split("_")[1]), []).append(path)
    return found


def chosen_slice(folder, patient):
    """A slice with a nodule mask if there is one, else the middle slice."""
    masks = masks_of(folder, patient)
    if masks:
        return sorted(masks)[len(masks) // 2]
    return n_slices(folder, patient) // 2


def load_slice(folder, patient, index):
    return np.load(folder / patient / f"slice_{index}.npy")


def load_mask(folder, patient, index):
    files = masks_of(folder, patient).get(index, [])
    mask = np.zeros((512, 512), dtype=bool)
    for path in files:
        mask |= np.load(path) > 0
    return mask


def extent(spacing, n=512):
    half = n * spacing / 2
    return (-half, half, half, -half)  # imshow: left, right, bottom, top (mm)


def body_stats(image, spacing):
    """Area (cm^2) and centroid (mm, row and column) of HU > -500."""
    body = image > -500
    area = body.sum() * spacing**2 / 100
    rows, cols = np.nonzero(body)
    centre = (np.array([rows.mean(), cols.mean()]) - 255.5) * spacing
    return area, centre


# --------------------------------------------------------------------------
# resampling


def check_resampling(args):
    folder, out = args.data, args.out
    spacings = original_spacings()
    present = [p for p in patients(folder) if p in spacings]
    if not present:
        raise SystemExit(f"No LIDC-IDRI patients in {folder}")
    by_spacing = sorted(present, key=lambda p: spacings[p])
    picks = [
        by_spacing[int(round(q * (len(by_spacing) - 1)))]
        for q in np.linspace(0, 1, args.n_patients)
    ]
    rows = []
    fig, axes = plt.subplots(len(picks), 4, figsize=(20, 4.6 * len(picks)))
    axes = np.atleast_2d(axes)
    print(
        "patient         spacing  FOV    body cm2 before/after  centre shift mm"
        "  nodule mm2 before/after"
    )
    for row, patient in zip(axes, picks):
        require_native(folder, patient)
        spacing = spacings[patient]
        index = chosen_slice(folder, patient)
        native = load_slice(folder, patient, index)
        new = resampling.resample_ct_slice(native, spacing)
        mask = load_mask(folder, patient, index)
        new_mask = resampling.resample_mask(mask.astype(np.int16), spacing) > 0
        # back onto the native grid: what resampling lost (smoothing, edges)
        back = resampling.resample_slice(new, NEW_SPACING, new_spacing=spacing)

        area0, centre0 = body_stats(native, spacing)
        area1, centre1 = body_stats(new, NEW_SPACING)
        nodule = (mask.sum() * spacing**2, new_mask.sum() * NEW_SPACING**2)
        shift = np.abs(centre1 - centre0).max()
        line = (
            f"{patient}  {spacing:.4f}  {512 * spacing:4.0f}   "
            f"{area0:7.1f} / {area1:7.1f}      {shift:5.2f}"
        )
        if mask.any():
            line += f"            {nodule[0]:7.1f} / {nodule[1]:7.1f}"
        print(line)
        rows.append(
            dict(
                patient=patient,
                slice=index,
                spacing_mm=spacing,
                body_cm2=[area0, area1],
                centre_shift_mm=shift,
                nodule_mm2=nodule if mask.any() else None,
            )
        )

        lim = (-252, 252)
        for ax, image, s, m, title in (
            (row[0], native, spacing, mask, f"before: {spacing:.3f} mm/pixel"),
            (row[1], new, NEW_SPACING, new_mask, f"after: {NEW_SPACING:.3f} mm/pixel"),
        ):
            ax.imshow(
                image,
                cmap="gray",
                vmin=HU_WINDOW[0],
                vmax=HU_WINDOW[1],
                extent=extent(s),
            )
            if m.any():
                ax.contour(
                    m,
                    levels=[0.5],
                    colors="r",
                    linewidths=0.8,
                    extent=(extent(s)[0], extent(s)[1], extent(s)[3], extent(s)[2]),
                )
            ax.set_xlim(lim)
            ax.set_ylim(lim[::-1])
            ax.set_title(f"{patient} slice {index}\n{title}", fontsize=9)
            ax.set_xlabel("mm")
        diff = row[2].imshow(
            native.astype(np.float32) - back,
            cmap="RdBu_r",
            vmin=-200,
            vmax=200,
            extent=extent(spacing),
        )
        row[2].set_title("before - after (back on the original grid), HU", fontsize=9)
        fig.colorbar(diff, ax=row[2], fraction=0.046)
        c0 = (np.arange(512) - 255.5) * spacing
        c1 = (np.arange(512) - 255.5) * NEW_SPACING
        row[3].plot(c0, native[256], label="before", lw=1)
        row[3].plot(c1, new[256], label="after", lw=1, ls="--")
        row[3].set_xlim(lim)
        row[3].set_xlabel("mm (row through the centre)")
        row[3].set_ylabel("HU")
        row[3].legend(fontsize=8)
    fig.tight_layout()
    path = out / "resampling.png"
    fig.savefig(path, dpi=80)
    (out / "resampling.json").write_text(json.dumps(rows, indent=2))
    print(f"Saved {path} and resampling.json")


# --------------------------------------------------------------------------
# doses


def experiments():
    from LION.experiments import ct_experiments

    return {
        "clinical": ct_experiments.clinicalCTRecon,
        "low dose": ct_experiments.LowDoseCTRecon,
        "extremely low dose": ct_experiments.ExtremeLowDoseCTRecon,
    }


def check_doses(args):
    import torch

    from LION.classical_algorithms import fdk
    from LION.CTtools import ct_utils as ct
    from LION.CTtools.ct_geometry import Geometry
    from LION.data_loaders.LIDC_IDRI import LIDC_IDRI
    from LION.metrics.psnr import PSNR

    folder, out, device = args.data, args.out, torch.device(args.device)
    torch.manual_seed(0)
    psnr = PSNR()
    spacings = original_spacings()
    old_op = ct.make_operator(Geometry.default_parameters())  # before #264
    new_op = ct.make_operator(LIDC_IDRI.default_geometry())  # after
    exps = {name: cls(cls.default_parameters()) for name, cls in experiments().items()}
    if args.sweep:
        for I0 in [float(v) for v in args.sweep.split(",")]:
            params = experiments()["clinical"].default_parameters()
            params.noise_params.I0 = I0
            exps[f"I0 {I0:g}"] = experiments()["clinical"](params)

    def density(hu):
        image = torch.from_numpy(hu.astype(np.float32))[None].to(device)
        return ct.from_HU_to_density(image)  # (1, 512, 512), as the loader

    # a small, a typical and a large patient: thickest water path, after
    candidates = [p for p in patients(folder) if p in spacings]
    candidates = candidates[:: max(1, len(candidates) // args.n_slices)]
    paths = []
    for patient in candidates:
        require_native(folder, patient)
        hu = load_slice(folder, patient, n_slices(folder, patient) // 2)
        new = resampling.resample_ct_slice(hu, spacings[patient])
        paths.append(new_op(density(new)).max().item() / 1.0137)  # mm of water
    paths = np.array(paths)
    picks = {}
    for label, q in (("small", 5), ("typical", 50), ("large", 95)):
        k = int(np.argmin(np.abs(paths - np.percentile(paths, q))))
        picks[label] = (candidates[k], paths[k])

    columns = []
    for name in exps:
        if name in experiments():
            columns += [(name, "before"), (name, "after")]
        else:
            columns += [(name, "after")]
    fig, axes = plt.subplots(
        len(picks),
        1 + len(columns),
        figsize=(3.4 * (1 + len(columns)), 3.8 * len(picks)),
    )
    for row, (label, (patient, water)) in zip(axes, picks.items()):
        spacing = spacings[patient]
        hu = load_slice(folder, patient, n_slices(folder, patient) // 2)
        images = {
            "before": (density(hu), old_op, spacing),
            "after": (
                density(resampling.resample_ct_slice(hu, spacing)),
                new_op,
                NEW_SPACING,
            ),
        }
        target, _, _ = images["after"]
        row[0].imshow(
            target[0].cpu().numpy(),
            cmap="gray",
            vmin=0,
            vmax=args.vmax,
            extent=extent(NEW_SPACING),
        )
        row[0].set_title(
            f"{label}: {patient}\n{spacing:.3f} mm, {water:.0f} mm water\n"
            "ground truth (after)",
            fontsize=8,
        )
        line = []
        for ax, (name, when) in zip(row[1:], columns):
            image, op, s = images[when]
            sino = exps[name].sino_fun(op(image))  # the experiment's noise
            recon = fdk(sino.unsqueeze(0), op)
            value = psnr(recon, image.unsqueeze(0)).item()
            line.append(f"{name} {when}: {value:.1f}")
            ax.imshow(
                recon[0, 0].cpu().numpy(),
                cmap="gray",
                vmin=0,
                vmax=args.vmax,
                extent=extent(s),
            )
            I0 = exps[name].param.noise_params.I0
            ax.set_title(
                f"{name}, {when} #264\nI0 {I0:g}, PSNR {value:.1f}", fontsize=8
            )
        for ax in row:
            ax.set_xlim(-252, 252)
            ax.set_ylim(252, -252)
            ax.set_xticks([])
            ax.set_yticks([])
        print(f"{label} patient {patient}, PSNR: " + "; ".join(line))
    fig.tight_layout()
    path = out / ("doses_sweep.png" if args.sweep else "doses.png")
    fig.savefig(path, dpi=90)
    print(f"Saved {path}")


# --------------------------------------------------------------------------
# trial and loader


def make_trial(args):
    source, trial = args.data, args.trial_dir
    target = trial / "processed" / "LIDC-IDRI"
    if target.exists():
        raise SystemExit(f"{target} exists; remove it or choose another --trial-dir")
    target.mkdir(parents=True)
    spacings = original_spacings()
    for path in source.glob("*.json"):
        shutil.copy2(path, target / path.name)
    expected = trial / "expected"
    expected.mkdir()
    chosen = [f"LIDC-IDRI-{i:04d}" for i in range(1, args.n_patients + 1)]
    for patient in chosen:
        if not (source / patient).is_dir():
            print(f"{patient} is not in {source}, skipping")
            continue
        require_native(source, patient)
        shutil.copytree(source / patient, target / patient)
        spacing = spacings[patient]
        n = n_slices(source, patient)
        for index in sorted({0, n // 2, n - 1}):
            np.save(
                expected / f"{patient}_slice_{index}.npy",
                resampling.resample_ct_slice(
                    load_slice(source, patient, index), spacing
                ),
            )
        for files in masks_of(source, patient).values():
            for path in files:
                np.save(
                    expected / f"{patient}_{path.name}",
                    resampling.resample_mask(np.load(path), spacing),
                )
    print(f"Copied {len(chosen)} patients to {target}; expected slices in {expected}.")
    print("Now convert the copy (pylidc needs setuptools<81, see README.md):")
    print(
        f"  LION_DATA_PATH={trial} python "
        f"{REPO}/LION/data_loaders/LIDC_IDRI/pre_process_lidc_idri.py --resample-only"
    )
    print(f"and then: python {__file__} loader --trial-dir {trial}")


def check_loader(args):
    import torch

    from LION.classical_algorithms import fdk
    from LION.CTtools import ct_utils as ct
    from LION.data_loaders.LIDC_IDRI import LIDC_IDRI

    trial = args.trial_dir
    folder = trial / "processed" / "LIDC-IDRI"
    out, problems = args.out, []

    # 1. what --resample-only wrote
    record = json.loads((folder / "pre_processing.json").read_text())
    print("pre_processing.json:", record)
    if record["pixel_spacing_mm"] != NEW_SPACING:
        problems.append("pre_processing.json has another spacing")
    spacings = original_spacings()
    for patient in patients(folder):
        if not is_resampled(folder, patient):
            problems.append(f"{patient} not resampled")
            continue
        saved = json.loads((folder / patient / "pixel_spacing.json").read_text())
        if saved["original_pixel_spacing_mm"] != spacings[patient]:
            problems.append(f"{patient}: original spacing {saved}")
    leftovers = [p.name for p in folder.iterdir() if p.name.endswith((".tmp", ".old"))]
    if leftovers:
        problems.append(f"temporary folders left: {leftovers}")
    compared = 0
    for path in sorted((trial / "expected").glob("*.npy")):
        patient, name = path.name[:14], path.name[15:]
        written = np.load(folder / patient / name)
        expected = np.load(path)
        if written.shape != (512, 512) or written.dtype != np.int16:
            problems.append(f"{patient}/{name}: {written.shape} {written.dtype}")
        elif not np.array_equal(written, expected):
            problems.append(f"{patient}/{name} differs from the expected slice")
        compared += 1
    print(f"Compared {compared} slices and masks with the expected ones.")

    # 2. the experiments through LION's data loader, on the trial copy
    torch.manual_seed(0)
    picks = []
    for name, cls in experiments().items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            experiment = cls(cls.default_parameters(), datafolder=folder)
            data = experiment.get_testing_dataset()
        for warning in caught:  # the loader's spacing and geometry warnings
            if "pixel" in str(warning.message):
                problems.append(f"{name}: {warning.message}")
        if not isinstance(data, LIDC_IDRI) or data.pixel_spacing_mm != NEW_SPACING:
            problems.append(f"{name}: the loader did not read the spacing")
        op = ct.make_operator(experiment.geometry)
        indices = np.linspace(0, len(data) - 1, args.n_slices).round().astype(int)
        for index in indices:
            sino, image = data[int(index)]
            if tuple(image.shape) != (1, 512, 512) or tuple(sino.shape[-1:]) != (1024,):
                problems.append(
                    f"{name}: shapes {tuple(sino.shape)} {tuple(image.shape)}"
                )
            picks.append(
                (name, int(index), image, fdk(sino.unsqueeze(0).to(image.device), op))
            )

    n_rows = len(experiments())
    fig, axes = plt.subplots(
        n_rows, 2 * args.n_slices, figsize=(3.2 * 2 * args.n_slices, 3.6 * n_rows)
    )
    for k, (name, index, image, recon) in enumerate(picks):
        r, c = divmod(k, args.n_slices)
        for ax, img, what in (
            (axes[r, 2 * c], image, "target"),
            (axes[r, 2 * c + 1], recon[0], "FDK"),
        ):
            ax.imshow(
                img[0].cpu().numpy(),
                cmap="gray",
                vmin=0,
                vmax=args.vmax,
                extent=extent(NEW_SPACING),
            )
            ax.set_title(f"{name}, test slice {index}: {what}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    path = out / "loader.png"
    fig.savefig(path, dpi=80)
    print(f"Saved {path}")
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print("  -", p)
        raise SystemExit(1)
    print("All checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("check", choices=["resampling", "doses", "trial", "loader"])
    parser.add_argument(
        "--data",
        type=pathlib.Path,
        default=None,
        help="processed LIDC-IDRI folder (default: LION's, from " "LION_DATA_PATH)",
    )
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("lidc_check"))
    parser.add_argument(
        "--n-patients",
        type=int,
        default=None,
        help="resampling: patients shown (default 6); trial: the "
        "first patients copied (default 30)",
    )
    parser.add_argument(
        "--n-slices",
        type=int,
        default=None,
        help="doses: patients looked at to pick the three "
        "(default 200); loader: test slices per experiment (default 3)",
    )
    parser.add_argument(
        "--sweep",
        default=None,
        help="doses: also the clinical experiment at these I0, "
        "comma-separated (e.g. 3e4,2e4,1e4,5e3,2e3,1e3), after #264",
    )
    parser.add_argument("--trial-dir", type=pathlib.Path, default=None)
    parser.add_argument(
        "--vmax", type=float, default=1.3, help="display window: 0 to vmax g/cm^3"
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.data is None and args.check != "loader":
        args.data = default_data_folder()
    if args.check in ("trial", "loader") and args.trial_dir is None:
        parser.error("trial and loader need --trial-dir")
    if args.n_patients is None:
        args.n_patients = 30 if args.check == "trial" else 6
    if args.n_slices is None:
        args.n_slices = 200 if args.check == "doses" else 3
    args.out.mkdir(parents=True, exist_ok=True)
    {
        "resampling": check_resampling,
        "doses": check_doses,
        "trial": make_trial,
        "loader": check_loader,
    }[args.check](args)
