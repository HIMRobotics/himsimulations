"""Shared fixtures for the physics tests: parameter sets independent of the priors file, so analytic tests stay put."""

from __future__ import annotations

import pytest

from lib.physics.params import BallTableParams, provisional_priors


@pytest.fixture
def base_params() -> BallTableParams:
    """The provisional nominal set."""

    return provisional_priors().nominal()


@pytest.fixture
def vacuum_params(base_params: BallTableParams) -> BallTableParams:
    """No air: gravity only between contacts."""

    return base_params.with_values({"air_density": 0.0})
