"""pytest fixtures for AEB scenario tests."""

import pytest

from core.aeb.calibration import AEBCalibration, DEFAULT as CAL_DEFAULT
from tests.aeb.harness import evaluate_frame


@pytest.fixture
def default_calibration() -> AEBCalibration:
    return CAL_DEFAULT


@pytest.fixture
def evaluator():
    return evaluate_frame
