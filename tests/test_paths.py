"""Path resolution."""

from lime_analytics.paths import reference_config_dir


def test_reference_config_exists() -> None:
    cfg = reference_config_dir() / "ftir_bands.yaml"
    assert cfg.is_file(), f"Expected {cfg} (run tests from repo with config/)"


def test_fesem_remarks_config_exists() -> None:
    cfg = reference_config_dir() / "fesem_remarks.yaml"
    assert cfg.is_file(), f"Expected {cfg}"


def test_fesem_wizard_config_exists() -> None:
    cfg = reference_config_dir() / "fesem_wizard.yaml"
    assert cfg.is_file(), f"Expected {cfg}"
