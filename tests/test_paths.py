"""Path resolution."""

from soil_analytics.paths import fesem_supervised_data_dir, reference_config_dir


def test_reference_config_exists() -> None:
    cfg = reference_config_dir() / "ftir_bands.yaml"
    assert cfg.is_file(), f"Expected {cfg} (run tests from repo with config/)"


def test_fesem_supervised_data_dir() -> None:
    d = fesem_supervised_data_dir()
    assert d.name == "fesem_supervised"
    assert d.parent.name == "data"
