"""Test padis readme warnings behaviour."""

from pathlib import Path


README = (
    Path(__file__).resolve().parents[2]
    / "scripts/paper_scripts/PaDIS-Reproduction/README.md"
)


def test_padis_readme_warns_about_lion_native_method_substitutions():
    """Verify that padis readme warns about lion native method substitutions."""
    text = README.read_text()
    warnings = text.split("## Notes and limitations", 1)[1]

    assert "baseline" in warnings
    assert "FDK" in warnings
    assert "FBP" in warnings
    assert "cp_tv" in warnings
    assert "Chambolle-Pock" in warnings
    assert "not the exact ADMM-TV algorithm described by Hu et al." in warnings
    assert "pnp_admm" in warnings
    assert "DRUNet denoiser" in warnings
    assert "LION-native DRUNet surrogate" in warnings
    assert "does not give enough optimizer, architecture, or" in warnings
    assert "stopping-rule detail" in warnings
    assert "no-PaDIS-prior" in warnings
    assert "empty" in warnings
    assert "checkpoint identity" in warnings
    assert "Slurm A100 path is an equivalent reproduction" in warnings
