# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""TIGRE against tomosipo, for LION's CT operator (internal: removed before the
PR is merged).

Run on a GPU machine with both backends installed:

    python scripts/tigre_vs_tomosipo.py --out tigre_vs_tomosipo

1. Conventions: which TIGRE angle sign, angle offset and detector direction
   give tomosipo's sinograms for the same LION geometry (fan and parallel).
   The best is used for the rest, and printed: if it isn't the one in
   ``LION/operators/ct/_tigre.py``, that file needs it.
2. Agreement, on LION's geometries: projections, backprojections, FDK (also
   against ``tigre.algorithms.fdk``/``fbp``), and each backend's adjointness.
3. Speed and memory over batch sizes: projection, backprojection, FDK, and
   one LPD training step (forward and backward).

Prints tables, and writes ``results.json`` and ``fdk.png`` in ``--out``.
"""

import argparse
import copy
import itertools
import json
import pathlib
import time

import numpy as np
import torch

from LION.operators.ct import CTProjectionOp, Geometry, _tigre, fdk


def phantom(shape, seed=0):
    """Random ellipses in a disc, (1, H, W), on the GPU."""
    rng = np.random.default_rng(seed)
    _, h, w = shape
    yy, xx = np.mgrid[-1 : 1 : h * 1j, -1 : 1 : w * 1j]
    img = 0.2 * (xx**2 + yy**2 < 0.8**2)
    for _ in range(12):
        cx, cy = rng.uniform(-0.5, 0.5, 2)
        a, b = rng.uniform(0.05, 0.3, 2)
        t = rng.uniform(0, np.pi)
        u = (xx - cx) * np.cos(t) + (yy - cy) * np.sin(t)
        v = -(xx - cx) * np.sin(t) + (yy - cy) * np.cos(t)
        img += rng.uniform(0.1, 0.5) * ((u / a) ** 2 + (v / b) ** 2 < 1)
    return torch.tensor(img[None], dtype=torch.float32, device="cuda")


def rel(a, b):
    return float(torch.linalg.norm(a - b) / torch.linalg.norm(b))


def with_backend(geometry, backend):
    g = copy.deepcopy(geometry)
    g.backend = backend
    return g


def small(mode):
    n = 128
    return Geometry(
        mode=mode,
        image_shape=[1, n, n],
        image_size=[1.0, float(n), float(n)],
        detector_shape=[1, 192],
        detector_size=[1.0, 192 * 1.5],
        dso=2.0 * n,
        dsd=4.0 * n,
        angles=np.linspace(0, 2 * np.pi, 180, endpoint=False),
    )


def conventions():
    """Rel. error of TIGRE's projections against tomosipo's for each
    convention; sets the best in ``_tigre``."""
    rows, best = [], {}
    defaults = (_tigre.ANGLE_SIGN, _tigre.ANGLE_OFFSET, _tigre.FLIP_U)
    for mode in ("fan", "parallel"):
        geometry = small(mode)
        x = phantom(geometry.image_shape)
        reference = CTProjectionOp(with_backend(geometry, "tomosipo"))(x)
        errors = {}
        for sign, offset, flip in itertools.product(
            (1.0, -1.0), (0.0, np.pi / 2, np.pi, -np.pi / 2), (False, True)
        ):
            _tigre.ANGLE_SIGN, _tigre.ANGLE_OFFSET, _tigre.FLIP_U = sign, offset, flip
            sino = CTProjectionOp(with_backend(geometry, "tigre"))(x)
            errors[(sign, offset, flip)] = rel(sino, reference)
            rows.append((mode, sign, offset, flip, errors[(sign, offset, flip)]))
        best[mode] = min(errors, key=errors.get)
    print("\n# Conventions (rel. error of TIGRE's sinogram against tomosipo's)")
    print(f"{'mode':9} {'sign':>5} {'offset':>8} {'flip_u':>6} {'rel err':>9}")
    for mode, sign, offset, flip, err in sorted(rows, key=lambda r: (r[0], r[4])):
        print(f"{mode:9} {sign:5.0f} {offset:8.4f} {str(flip):>6} {err:9.2e}")
    if best["fan"] != best["parallel"]:
        print(f"!! fan and parallel disagree: {best}")
    chosen = best["fan"]
    print(f"best: ANGLE_SIGN={chosen[0]}, ANGLE_OFFSET={chosen[1]}, FLIP_U={chosen[2]}")
    print(f"in _tigre.py: {defaults}" + ("" if chosen == defaults else "  <-- CHANGE"))
    _tigre.ANGLE_SIGN, _tigre.ANGLE_OFFSET, _tigre.FLIP_U = chosen
    return {"rows": rows, "best": {k: list(v) for k, v in best.items()}}


def adjointness(op, x):
    y = torch.rand(op.range_shape, device="cuda")
    lhs, rhs = float((op(x) * y).sum()), float((x * op.adjoint(y)).sum())
    return abs(lhs - rhs) / abs(lhs)


def agreement(out):
    tigre = __import__("tigre")
    geometries = {
        "fan default (512, 360 angles)": Geometry.default_parameters(),
        "fan sparse view (50 angles)": Geometry.sparse_view_parameters(),
        "fan limited angle (60 deg)": Geometry.sparse_angle_parameters(),
        "parallel default (512)": Geometry.parallel_default_parameters(),
    }
    results, images = {}, {}
    print("\n# Agreement (relative errors)")
    header = (
        "geometry",
        "fp",
        "bp",
        "bp scale",
        "fdk",
        "fdk vs x ts",
        "fdk vs x tigre",
        "vs tigre.alg",
        "adj ts",
        "adj tigre",
    )
    print(" | ".join(header))
    for name, geometry in geometries.items():
        ts_op = CTProjectionOp(with_backend(geometry, "tomosipo"))
        tg_op = CTProjectionOp(with_backend(geometry, "tigre"))
        x = phantom(geometry.image_shape)
        ts_sino, tg_sino = ts_op(x), tg_op(x)
        y = torch.rand(ts_op.range_shape, device="cuda")
        ts_bp, tg_bp = ts_op.adjoint(y), tg_op.adjoint(y)
        ts_fdk, tg_fdk = fdk(ts_sino, ts_op), fdk(ts_sino, tg_op)
        # TIGRE's own FDK/FBP, on the same (tomosipo) sinogram
        proj = ts_sino.permute(1, 0, 2)
        if _tigre.FLIP_U:
            proj = proj.flip(-1)
        proj = proj.contiguous().cpu().numpy()
        algorithm = (
            tigre.algorithms.fdk if geometry.mode == "fan" else tigre.algorithms.fbp
        )
        kwargs = {"parker": False} if geometry.mode == "fan" else {}
        tg_alg = algorithm(proj, tg_op._tigre_geo, tg_op._tigre_angles, **kwargs)
        tg_alg = torch.clip(torch.from_numpy(tg_alg).cuda(), min=0)
        r = {
            "fp": rel(tg_sino, ts_sino),
            "bp": rel(tg_bp, ts_bp),
            "bp scale (tigre/ts)": float(tg_bp.norm() / ts_bp.norm()),
            "fdk": rel(tg_fdk, ts_fdk),
            "fdk vs x (tomosipo)": rel(ts_fdk, x),
            "fdk vs x (tigre)": rel(tg_fdk, x),
            "fdk vs tigre.algorithms": rel(tg_fdk, tg_alg),
            "adjointness tomosipo": adjointness(ts_op, x),
            "adjointness tigre": adjointness(tg_op, x),
        }
        results[name] = r
        images[name] = [
            t.cpu().numpy()[0] for t in (x, ts_fdk, tg_fdk, tg_fdk - ts_fdk)
        ]
        print(name, "|", " | ".join(f"{v:.2e}" for v in r.values()))
    figure(images, out / "fdk.png")
    return results


def figure(images, path):
    from matplotlib.figure import Figure

    fig = Figure(figsize=(16, 4 * len(images)))
    axes = fig.subplots(len(images), 4, squeeze=False)
    for row, (name, imgs) in zip(axes, images.items()):
        for ax, img, title in zip(
            row, imgs, ("x", "FDK tomosipo", "FDK TIGRE", "TIGRE - tomosipo")
        ):
            im = ax.imshow(img, cmap="gray" if "-" not in title else "RdBu")
            ax.set_title(f"{name}: {title}", fontsize=8)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(path, dpi=100, bbox_inches="tight")


def timed(fn, repeats=5):
    fn()  # warm up
    torch.cuda.synchronize()
    times = []
    torch.cuda.reset_peak_memory_stats()
    free_before = torch.cuda.mem_get_info()[0]
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return {
        "median_s": float(np.median(times)),
        "torch_peak_MB": torch.cuda.max_memory_allocated() / 2**20,
        "free_change_MB": (free_before - torch.cuda.mem_get_info()[0]) / 2**20,
    }


def speed(batch_sizes):
    from LION.models.iterative_unrolled.LPD import LPD

    results = {}
    geometry = Geometry.default_parameters()
    print(
        "\n# Speed (median s) and torch's peak memory (MB; TIGRE's own CUDA "
        "allocations aren't in it)"
    )
    print("backend | batch | fp | bp | fdk | LPD step")
    for backend in ("tomosipo", "tigre"):
        op = CTProjectionOp(with_backend(geometry, backend))
        A, AT = op.autograd()
        model = LPD(with_backend(geometry, backend)).cuda()
        optimiser = torch.optim.Adam(model.parameters())
        for b in batch_sizes:
            x = torch.stack([phantom(geometry.image_shape, s) for s in range(b)])
            y = A(x).detach()

            def lpd_step():
                optimiser.zero_grad()
                loss = torch.nn.functional.mse_loss(model(y), x)
                loss.backward()
                optimiser.step()

            r = {
                "fp": timed(lambda: A(x)),
                "bp": timed(lambda: AT(y)),
                "fdk": timed(lambda: fdk(y, op)),
                "LPD step": timed(lpd_step, repeats=3),
            }
            results[f"{backend}, batch {b}"] = r
            print(
                backend,
                "|",
                b,
                "|",
                " | ".join(
                    f"{v['median_s']:.3f} ({v['torch_peak_MB']:.0f})"
                    for v in r.values()
                ),
            )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=pathlib.Path, default=pathlib.Path("tigre_vs_tomosipo")
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = {
        "conventions": conventions(),
        "agreement": agreement(args.out),
        "speed": speed(args.batch_sizes),
        "gpu": torch.cuda.get_device_name(),
    }
    with open(args.out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwritten {args.out / 'results.json'}, {args.out / 'fdk.png'}")


if __name__ == "__main__":
    main()
