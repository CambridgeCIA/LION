# PCM GUI guide

This guide is intentionally short while the interface is still changing. The GUI Help window reads this same file, so explanations do not need to be maintained twice.

## Quick start

1. Choose a **Preset**. For quick GUI testing, `cigs_example_256x256` is the small 256 x 256 example.
2. Choose a reconstruction method and device.
3. Adjust acquisition/sampling and output settings if needed.
4. Press **Run reconstruction**.
5. Use **PnP iteration analysis...** for method comparisons, parameter variants, iteration snapshots and custom plots.

> **SCREENSHOT TODO:** Main PCM window showing Ground truth | Simulated acquisition | Reconstruction.

## Acquisition and sampling

The acquisition panel controls sampling, simulated noise and display-only illumination choices. The green projected pattern and optional bias-light illustration do not alter the actual simulated measurement model.

> **SCREENSHOT TODO:** Acquisition settings and acquisition preview with progress/ETA.

## Reconstruction methods

Each PnP iteration-analysis method card is independent. The method type can be changed from its dropdown. Use **Settings...** to edit the method dataclass, **Duplicate** to make a hand-picked variant, and **Up/Down** to reorder methods.

Sweeps are optional. A sweep expands the Cartesian product only within that method card. For a small number of hand-picked parameter combinations, duplicate a card instead.

> **SCREENSHOT TODO:** Two method cards with different PnP-ADMM parameters.

## Save and reload a PnP iteration analysis

Use **Export YAML...** before running to save the complete analysis setup. Use **Import YAML...** later to restore it and continue editing.

When an analysis is run, the same re-importable definition is automatically saved as `analysis.yaml` in the analysis result folder.

The YAML stores the preset, sampling settings, noise scenarios, plot/snapshot choices, ordered method configurations and parameter sweeps. It is intended as a research notebook artifact, not just an internal cache.

> **SCREENSHOT TODO:** PnP iteration-analysis window with Import YAML / Export YAML buttons.

## Snapshots and stopping points

Snapshots may be selected explicitly or periodically, for example every 5 accepted iterations. Optional special snapshots include best, last, GS-PnP early-stop and GS-PnP backtracking-stop points.

## Custom Plot

Custom Plot can overlay any completed reconstructions, choose metrics, switch between iteration and reconstruction-time axes, edit legend labels, adjust display/save DPI and save PNG/PDF/SVG output. Plot display choices are remembered between openings.

> **SCREENSHOT TODO:** Custom Plot window with several reconstruction curves and a common outside legend.

## Field help

Method forms are generated from dataclasses. A `?` button appears when help is available from either:

- `dataclasses.field(metadata={"help": "..."})`, or
- the dataclass `Attributes:` docstring section.

This is deliberate: help text should live with the configuration definition so the same source can later feed GUI tooltips, generated docs and CLI documentation.

## Measurement files and live reconstruction

**TODO:** Add the real acquisition-file adapter after a representative hardware `.txt` file is available. The expected hardware format has two columns (pattern index and measured current), but the exact pattern-index convention and whether positive/complement patterns are separate must be confirmed before connecting it to reconstruction.

The intended architecture is:

```text
Acquisition -> measurement file/stream -> reconstruction A
                                    +--> reconstruction B
                                    +--> reconstruction C
```

This will allow acquisition to happen once while reconstruction methods can be changed or rerun later.

## Screenshot checklist

- [ ] Main PCM window
- [ ] Acquisition settings and preview
- [ ] PnP iteration-analysis window
- [ ] Method settings with `?` help
- [ ] Snapshot viewer
- [ ] Custom Plot builder
- [ ] Imported/exported analysis YAML workflow
