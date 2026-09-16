"""Sanity checks for the static IBC rule engine."""

from app.decisional.rule_engine import classify_caudal_spread


def test_ideal_caudal_spread():
    assert classify_caudal_spread(180.0) == "Ideal"


def test_disqualify_bands():
    assert classify_caudal_spread(215.0) == "Disqualify"
    assert classify_caudal_spread(160.0) == "Disqualify"


def test_slight_fault_band():
    assert classify_caudal_spread(190.0) == "Slight Fault"
