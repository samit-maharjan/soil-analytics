"""Path resolution."""

from soil_analytics.paths import reference_config_dir


def test_reference_config_exists() -> None:
    cfg = reference_config_dir() / "ftir_bands.yaml"
    assert cfg.is_file(), f"Expected {cfg} (run tests from repo with config/)"
