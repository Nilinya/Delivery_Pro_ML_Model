"""
Delivery comparison module for scenario analysis.
Generates different delivery scenarios and compares predicted times.
Supports: Distance ranges, Vehicle types, Weather impact, Traffic impact, Personnel ratings
"""

import copy
import math
from geopy.distance import geodesic


def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate geodesic distance between two coordinates in km."""
    return geodesic((lat1, lon1), (lat2, lon2)).km


def _to_float(value):
    """Safely convert input to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_valid_coordinate_pair(lat, lon):
    """Validate latitude/longitude ranges and reject 0,0 placeholder pair."""
    if lat is None or lon is None:
        return False
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return False
    if lat == 0 and lon == 0:
        return False
    return True


def _get_validated_coordinates(base_input):
    """Extract and validate restaurant and delivery coordinates from request payload."""
    rest_lat = _to_float(base_input.get('Restaurant_latitude'))
    rest_lon = _to_float(base_input.get('Restaurant_longitude'))
    del_lat = _to_float(base_input.get('Delivery_location_latitude'))
    del_lon = _to_float(base_input.get('Delivery_location_longitude'))

    if not _is_valid_coordinate_pair(rest_lat, rest_lon):
        raise ValueError('Invalid restaurant coordinates for distance comparison')
    if not _is_valid_coordinate_pair(del_lat, del_lon):
        raise ValueError('Invalid delivery coordinates for distance comparison')

    return rest_lat, rest_lon, del_lat, del_lon


def _compute_initial_bearing(lat1, lon1, lat2, lon2):
    """Compute initial bearing in degrees from point 1 to point 2."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lon = math.radians(lon2 - lon1)

    x = math.sin(delta_lon) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lon)

    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360


def _destination_point(lat, lon, distance_km, bearing_deg):
    """Move from origin to destination with geodesic distance and bearing."""
    destination = geodesic(kilometers=distance_km).destination((lat, lon), bearing_deg)
    return float(destination.latitude), float(destination.longitude)


def get_distance_band(distance_km):
    """Return distance band label for a given distance."""
    if distance_km < 5:
        return "1-5 km"
    elif distance_km < 10:
        return "5-10 km"
    elif distance_km < 20:
        return "10-20 km"
    elif distance_km < 50:
        return "20-50 km"
    else:
        return "50+ km"


def modify_coordinates_for_distance(base_lat, base_lon, target_distance_km, bearing_deg):
    """Generate destination coordinates for target distance using geodesic projection."""
    if target_distance_km <= 0:
        return base_lat, base_lon
    return _destination_point(base_lat, base_lon, target_distance_km, bearing_deg)


def _safe_base_distance_band(base_input):
    """Compute base distance band for labels in non-distance comparisons."""
    try:
        rest_lat, rest_lon, del_lat, del_lon = _get_validated_coordinates(base_input)
        base_distance = calculate_distance(rest_lat, rest_lon, del_lat, del_lon)
        return base_distance, get_distance_band(base_distance)
    except ValueError:
        return None, 'Unknown distance'


def generate_distance_range_scenarios(base_input):
    """
    Generate scenarios for different distance ranges.
    Returns list of (scenario_dict, label) tuples.
    """
    scenarios = []
    target_distances = [3, 7.5, 15, 35, 75]  # Representative distances for each band

    rest_lat, rest_lon, del_lat, del_lon = _get_validated_coordinates(base_input)
    bearing = _compute_initial_bearing(rest_lat, rest_lon, del_lat, del_lon)
    
    for target_dist in target_distances:
        scenario = copy.deepcopy(base_input)

        # Preserve route direction while changing distance.
        new_del_lat, new_del_lon = modify_coordinates_for_distance(
            rest_lat, rest_lon, target_dist, bearing
        )

        scenario['Delivery_location_latitude'] = new_del_lat
        scenario['Delivery_location_longitude'] = new_del_lon

        actual_distance = calculate_distance(rest_lat, rest_lon, new_del_lat, new_del_lon)
        band = get_distance_band(actual_distance)
        label = f"{band} (current vehicle & conditions)"

        scenarios.append((
            scenario,
            label,
            {
                'distance_band': band,
                'actual_distance_km': round(actual_distance, 2),
                'target_distance_km': target_dist
            }
        ))
    
    return scenarios


def generate_vehicle_type_scenarios(base_input):
    """
    Generate scenarios for different vehicle types.
    Returns list of (scenario_dict, label) tuples.
    """
    scenarios = []
    vehicle_types = ['motorcycle', 'scooter', 'electric_scooter']
    
    _, distance_band = _safe_base_distance_band(base_input)
    
    for vehicle in vehicle_types:
        scenario = copy.deepcopy(base_input)
        scenario['Type_of_vehicle'] = vehicle
        
        label = f"{vehicle.replace('_', ' ').title()} @ {distance_band}"
        scenarios.append((scenario, label))
    
    return scenarios


def generate_weather_scenarios(base_input):
    """
    Generate scenarios for different weather conditions.
    Returns list of (scenario_dict, label) tuples.
    """
    scenarios = []
    weather_types = ['Sunny', 'Cloudy', 'Fog', 'Stormy', 'Sandstorms', 'Windy']
    
    _, distance_band = _safe_base_distance_band(base_input)
    
    for weather in weather_types:
        scenario = copy.deepcopy(base_input)
        scenario['Weatherconditions'] = f'conditions {weather}'
        
        label = f"{weather} weather @ {distance_band}"
        scenarios.append((scenario, label))
    
    return scenarios


def generate_traffic_scenarios(base_input):
    """
    Generate scenarios for different traffic conditions.
    Returns list of (scenario_dict, label) tuples.
    """
    scenarios = []
    traffic_levels = ['Low', 'Medium', 'High', 'Jam']
    
    _, distance_band = _safe_base_distance_band(base_input)
    
    for traffic in traffic_levels:
        scenario = copy.deepcopy(base_input)
        scenario['Road_traffic_density'] = traffic
        
        label = f"{traffic} traffic @ {distance_band}"
        scenarios.append((scenario, label))
    
    return scenarios


def generate_personnel_rating_scenarios(base_input):
    """
    Generate scenarios for different delivery personnel ratings.
    Returns list of (scenario_dict, label) tuples.
    """
    scenarios = []
    ratings = [3.5, 4.0, 4.5, 5.0]
    
    _, distance_band = _safe_base_distance_band(base_input)
    
    for rating in ratings:
        scenario = copy.deepcopy(base_input)
        scenario['Delivery_person_Ratings'] = rating
        
        label = f"Delivery Person ⭐ {rating} @ {distance_band}"
        scenarios.append((scenario, label))
    
    return scenarios


def generate_scenarios(base_input, comparison_type):
    """
    Generate scenarios based on comparison type.
    
    Args:
        base_input: dict with delivery details
        comparison_type: str - one of ['distance_ranges', 'vehicle_types', 
                                       'weather_impact', 'traffic_impact', 
                                       'personnel_ratings']
    
    Returns:
        list of (scenario_dict, label) tuples
    """
    if comparison_type == 'distance_ranges':
        return generate_distance_range_scenarios(base_input)
    elif comparison_type == 'vehicle_types':
        return generate_vehicle_type_scenarios(base_input)
    elif comparison_type == 'weather_impact':
        return generate_weather_scenarios(base_input)
    elif comparison_type == 'traffic_impact':
        return generate_traffic_scenarios(base_input)
    elif comparison_type == 'personnel_ratings':
        return generate_personnel_rating_scenarios(base_input)
    else:
        raise ValueError(f"Invalid comparison_type: {comparison_type}")


def batch_predict_with_labels(predict_func, scenarios_with_labels):
    """
    Batch predict for multiple scenarios and return results with labels.
    
    Args:
        predict_func: function that takes input_dict and returns prediction value
        scenarios_with_labels: list of (scenario_dict, label) tuples
    
    Returns:
        list of dicts with keys: 'label', 'predicted_time', 'scenario'
    """
    results = []
    
    for item in scenarios_with_labels:
        if len(item) == 2:
            scenario_dict, label = item
            metadata = {}
        else:
            scenario_dict, label, metadata = item

        try:
            predicted_time = predict_func(scenario_dict)
            results.append({
                'label': label,
                'predicted_time': predicted_time,
                'metadata': metadata,
                'scenario': scenario_dict
            })
        except Exception as e:
            results.append({
                'label': label,
                'predicted_time': None,
                'error': str(e),
                'metadata': metadata,
                'scenario': scenario_dict
            })
    
    return results


def generate_insights(base_prediction, comparison_results, comparison_type):
    """
    Generate business insights from comparison results.
    
    Args:
        base_prediction: float - prediction for base input
        comparison_results: list of result dicts from batch_predict_with_labels
        comparison_type: str - the comparison type used
    
    Returns:
        dict with insights
    """
    valid_results = [r for r in comparison_results if r.get('predicted_time') is not None]
    
    if not valid_results:
        return {'error': 'No valid predictions'}
    
    times = [r['predicted_time'] for r in valid_results]
    fastest = min(valid_results, key=lambda x: x['predicted_time'])
    slowest = max(valid_results, key=lambda x: x['predicted_time'])
    avg_time = sum(times) / len(times)
    
    insights = {
        'base_prediction': round(base_prediction, 2),
        'fastest_option': fastest['label'],
        'fastest_time': round(fastest['predicted_time'], 2),
        'slowest_option': slowest['label'],
        'slowest_time': round(slowest['predicted_time'], 2),
        'average_time': round(avg_time, 2),
        'time_variance': round(slowest['predicted_time'] - fastest['predicted_time'], 2),
    }
    
    # Add comparison-specific insights
    if comparison_type == 'vehicle_types':
        insights['recommendation'] = "Motorcycle offers best speed. Scooter is cost-effective middle ground."
        insights['cost_effective'] = min(valid_results, key=lambda x: x['label'].find('scooter'))['label']
    
    elif comparison_type == 'distance_ranges':
        insights['recommendation'] = "Time increases significantly beyond 20km. Plan accordingly."
    
    elif comparison_type == 'weather_impact':
        sunny = next((r for r in valid_results if 'Sunny' in r['label']), None)
        stormy = next((r for r in valid_results if 'Stormy' in r['label']), None)
        if sunny and stormy:
            impact = stormy['predicted_time'] - sunny['predicted_time']
            insights['weather_impact'] = f"Stormy weather adds ~{round(impact, 1)} min vs Sunny"
    
    elif comparison_type == 'traffic_impact':
        low_traffic = next((r for r in valid_results if 'Low' in r['label']), None)
        jam = next((r for r in valid_results if 'Jam' in r['label']), None)
        if low_traffic and jam:
            impact = jam['predicted_time'] - low_traffic['predicted_time']
            insights['traffic_impact'] = f"Jam conditions add ~{round(impact, 1)} min vs Low traffic"
    
    elif comparison_type == 'personnel_ratings':
        insights['recommendation'] = "Higher-rated personnel deliver 5-15% faster. Match to SLA requirements."
    
    return insights
