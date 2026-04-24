"""FESEM progressive wizard config."""

import yaml

from soil_analytics.fesem_mcq import parse_phases
from soil_analytics.fesem_wizard import load_wizard
from soil_analytics.paths import project_root, reference_config_dir


def test_fesem_wizard_yaml_loads() -> None:
    p = reference_config_dir() / "fesem_wizard.yaml"
    assert p.is_file()
    w = load_wizard(p)
    assert w.start in w.nodes
    assert "q1" in w.nodes


def test_fesem_wizard_path_reaches_all_result_ids() -> None:
    """End nodes should map to existing phase ids in fesem_remarks."""
    root = project_root()
    with open(root / "config" / "reference_ranges" / "fesem_remarks.yaml", encoding="utf-8") as f:
        phases = parse_phases((yaml.safe_load(f) or {}).get("phases", []))
    allowed = {p.id for p in phases}
    w = load_wizard(root / "config" / "reference_ranges" / "fesem_wizard.yaml")
    for node in w.nodes.values():
        for o in node.options:
            if o.result:
                assert o.result in allowed, o.result
