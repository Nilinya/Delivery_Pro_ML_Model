"""Tests for comparison scenario generation logic."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from comparison import (
    calculate_distance,
    generate_distance_range_scenarios,
    generate_vehicle_type_scenarios,
)


def _sample_base_payload():
    return {
        'Delivery_person_Age': 34,
        'Delivery_person_Ratings': 4.5,
        'Restaurant_latitude': 12.913041,
        'Restaurant_longitude': 77.683237,
        'Delivery_location_latitude': 13.043041,
        'Delivery_location_longitude': 77.813237,
        'Weatherconditions': 'conditions Stormy',
        'Road_traffic_density': 'Jam',
        'Vehicle_condition': 2,
        'Type_of_order': 'Snack',
        'Type_of_vehicle': 'scooter',
        'multiple_deliveries': 1,
        'Festival': 'No',
        'City': 'Metropolitian',
        'Order_Date': '25-03-2022',
        'Time_Orderd': '19:45:00',
        'Time_Order_picked': '19:50:00',
    }


def test_distance_range_scenarios_target_distance_tolerance():
    payload = _sample_base_payload()
    scenarios = generate_distance_range_scenarios(payload)
    expected_targets = [3, 7.5, 15, 35, 75]

    assert len(scenarios) == len(expected_targets)

    rest_lat = payload['Restaurant_latitude']
    rest_lon = payload['Restaurant_longitude']

    for scenario_tuple, target in zip(scenarios, expected_targets):
        scenario, _label, metadata = scenario_tuple
        actual = calculate_distance(
            rest_lat,
            rest_lon,
            scenario['Delivery_location_latitude'],
            scenario['Delivery_location_longitude'],
        )
        assert abs(actual - target) <= 0.5
        assert math.isclose(metadata['actual_distance_km'], round(actual, 2), rel_tol=0, abs_tol=0.01)


def test_distance_range_scenarios_invalid_coordinates_raise_value_error():
    payload = _sample_base_payload()
    payload['Restaurant_latitude'] = 0.0
    payload['Restaurant_longitude'] = 0.0

    with pytest.raises(ValueError):
        generate_distance_range_scenarios(payload)


def test_vehicle_scenarios_do_not_override_vehicle_condition():
    payload = _sample_base_payload()
    payload['Vehicle_condition'] = 0

    scenarios = generate_vehicle_type_scenarios(payload)

    assert len(scenarios) == 3
    for scenario, _label in scenarios:
        assert scenario['Vehicle_condition'] == 0
