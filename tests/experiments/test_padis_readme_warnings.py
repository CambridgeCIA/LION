from pathlib import Path


README = (
    Path(__file__).resolve().parents[2]
    / "scripts/paper_scripts/PaDIS-Reproduction/README.md"
)


def test_padis_readme_warns_about_lion_native_method_substitutions():
    text = README.read_text()
    warnings = text.split("## Notes And Warnings", 1)[1]

    assert "baseline" in warnings
    assert "FDK" in warnings
    assert "FBP" in warnings
    assert "admm_tv" in warnings
    assert "Chambolle-Pock" in warnings
    assert "not the paper's exact ADMM-TV" in warnings
    assert "pnp_admm" in warnings
    assert "DRUNet denoiser" in warnings
    assert "LION-native DRUNet surrogate" in warnings
    assert "does not give enough optimizer, architecture, or" in warnings
    assert "stopping-rule detail" in warnings
    assert "no-PaDIS-prior" in warnings
    assert "empty" in warnings
    assert "checkpoint identity" in warnings
    assert "A100/CUDA CT validation" in warnings
    assert "public PaDIS repository only provides" in warnings
