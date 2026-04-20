"""Path resolution."""

from soil_analytics.paths import fesem_supervised_data_dir, reference_config_dir


def test_reference_config_exists() -> None:
    cfg = reference_config_dir() / "ftir_bands.yaml"
    assert cfg.is_file(), f"Expected {cfg} (run tests from repo with config/)"


def test_fesem_supervised_data_dir() -> None:
    d = fesem_supervised_data_dir()
    assert d.name == "fesem_supervised"


def test_fesem_subdirs_helpers() -> None:
    from soil_analytics.paths import fesem_analysis_dir, fesem_micrographs_dir

    assert fesem_micrographs_dir().name == "micrographs"
    assert fesem_analysis_dir().name == "analysis"
    assert fesem_micrographs_dir().parent == fesem_supervised_data_dir()
    assert fesem_supervised_data_dir().parent.name == "data"


def test_fesem_remarks_config_exists() -> None:
    cfg = reference_config_dir() / "fesem_remarks.yaml"
    assert cfg.is_file(), f"Expected {cfg}"
