"""Canonical data models for spectroscopy and thermal curves."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class FTIRSeries(BaseModel):
    """FTIR spectrum: x = wavenumber (cm^-1), y = absorbance or transmittance."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    wavenumber_cm1: np.ndarray
    y: np.ndarray
    y_label: Literal["absorbance", "transmittance", "reflectance", "unknown"] = "unknown"
    source_name: str | None = None

    @model_validator(mode="after")
    def _check_shapes(self) -> FTIRSeries:
        if self.wavenumber_cm1.shape != self.y.shape:
            raise ValueError("wavenumber_cm1 and y must have the same shape")
        return self


class XRDPattern(BaseModel):
    """XRD pattern: 2theta (degrees), intensity (counts or arbitrary)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    two_theta_deg: np.ndarray
    intensity: np.ndarray
    source_name: str | None = None

    @model_validator(mode="after")
    def _check_shapes(self) -> XRDPattern:
        if self.two_theta_deg.shape != self.intensity.shape:
            raise ValueError("two_theta_deg and intensity must have the same shape")
        return self


class TGACurve(BaseModel):
    """TGA: temperature (°C), mass; optional DTG."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    temperature_c: np.ndarray
    mass: np.ndarray
    mass_label: str = Field(default="mass", description="e.g. mass_mg, mass_fraction")
    dtg: np.ndarray | None = None
    source_name: str | None = None

    @model_validator(mode="after")
    def _check_shapes(self) -> TGACurve:
        if self.temperature_c.shape != self.mass.shape:
            raise ValueError("temperature_c and mass must have the same shape")
        if self.dtg is not None and self.dtg.shape != self.mass.shape:
            raise ValueError("dtg must match mass length")
        return self
