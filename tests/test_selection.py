from __future__ import annotations

from byakugan_app.models.selection import PixelSelection, PointMeasurement


def test_point_measurement_serialize_rounds_to_six_decimals_and_includes_quality():
    measurement = PointMeasurement(
        pixel=PixelSelection(u=13, v=21, width=200, height=100),
        depth_m=12.123456789,
        enu_vector=(1.234567891, -2.345678912, 3.456789123),
        geodetic=(6.622974148, 3.364868734, 54.354436211),
        quality_score=83.67,
        quality_label="Good",
    )

    payload = measurement.serialize()
    assert payload["depth_m"] == 12.123457
    assert payload["enu_e"] == 1.234568
    assert payload["enu_n"] == -2.345679
    assert payload["enu_u"] == 3.456789
    assert payload["latitude"] == 6.622974
    assert payload["longitude"] == 3.364869
    assert payload["altitude"] == 54.354436
    assert payload["quality_score"] == 83.67
    assert payload["quality_label"] == "Good"
