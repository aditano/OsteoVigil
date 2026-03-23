from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

MODULE_NAME = "cpt_predictor.materials"


if importlib.util.find_spec(MODULE_NAME) is None:
    pytest.skip("materials module not yet implemented", allow_module_level=True)

materials = importlib.import_module(MODULE_NAME)


def _get_first_attr(names: list[str]):
    for name in names:
        if hasattr(materials, name):
            return getattr(materials, name)
    raise AttributeError(f"None of the expected attributes were found: {names}")


def test_material_mapping_exposes_monotonic_properties():
    mapper = _get_first_attr(
        [
            "hu_to_bone_properties",
            "map_hu_to_properties",
            "hu_to_properties",
            "bone_properties_from_hu",
            "compute_bone_properties",
        ]
    )

    low = mapper(-500)
    mid = mapper(500)
    high = mapper(1500)

    def extract(field_names: list[str], payload):
        if isinstance(payload, dict):
            for key in field_names:
                if key in payload:
                    return float(payload[key])
        for key in field_names:
            if hasattr(payload, key):
                return float(getattr(payload, key))
        raise AssertionError(f"Could not extract any of {field_names} from {payload!r}")

    density_field = [
        "density",
        "rho",
        "rho_app",
        "apparent_density",
        "density_g_cm3",
    ]
    modulus_field = [
        "youngs_modulus_mpa",
        "youngs_modulus",
        "E_MPa",
        "E",
        "modulus_mpa",
    ]
    strength_field = [
        "yield_strength_mpa",
        "strength_mpa",
        "sigma_y_mpa",
        "sigma_y",
        "ultimate_strength_mpa",
    ]

    low_density = extract(density_field, low)
    mid_density = extract(density_field, mid)
    high_density = extract(density_field, high)

    low_modulus = extract(modulus_field, low)
    mid_modulus = extract(modulus_field, mid)
    high_modulus = extract(modulus_field, high)

    low_strength = extract(strength_field, low)
    mid_strength = extract(strength_field, mid)
    high_strength = extract(strength_field, high)

    assert low_density <= mid_density <= high_density
    assert low_modulus <= mid_modulus <= high_modulus
    assert low_strength <= mid_strength <= high_strength
    assert np.isfinite([low_density, mid_density, high_density]).all()
    assert np.isfinite([low_modulus, mid_modulus, high_modulus]).all()
    assert np.isfinite([low_strength, mid_strength, high_strength]).all()


def test_hu_to_bone_properties_accepts_scalar_input():
    mapper = _get_first_attr(
        [
            "hu_to_bone_properties",
            "map_hu_to_properties",
            "hu_to_properties",
        ]
    )

    sig = inspect.signature(mapper)
    assert len(sig.parameters) >= 1

    result = mapper(750)
    assert result is not None

    if isinstance(result, dict):
        assert any(
            key in result
            for key in [
                "density",
                "density_g_cm3",
                "youngs_modulus_mpa",
                "yield_strength_mpa",
            ]
        )


def test_material_lookup_table_is_sorted_and_covering():
    table = None
    for name in [
        "MATERIAL_BINS",
        "MATERIAL_LUT",
        "HU_MATERIAL_MAP",
        "MATERIAL_MAP",
        "material_bins",
    ]:
        if hasattr(materials, name):
            table = getattr(materials, name)
            break

    if table is None:
        pytest.skip("material bin lookup table not exposed by implementation")

    if isinstance(table, dict):
        items = list(table.items())
        keys = [float(k) for k, _ in items]
        assert keys == sorted(keys)
        assert len(items) >= 3
    else:
        entries = list(table)
        assert len(entries) >= 3
        thresholds = []
        for entry in entries:
            if isinstance(entry, dict):
                thresholds.append(float(entry.get("hu", entry.get("threshold", 0.0))))
            elif isinstance(entry, (tuple, list)) and entry:
                thresholds.append(float(entry[0]))
        assert thresholds == sorted(thresholds)
