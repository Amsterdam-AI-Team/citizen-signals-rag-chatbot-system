"""Geo utils module for transformations between addresses and coordinate systems"""
import requests
from pyproj import Transformer

import config as cfg


def wgs84_to_rd(longitude, latitude):
    """
    Converts WGS84 coordinates to RD New (EPSG:28992) coordinates.

    Args:
        longitude (float): Longitude in WGS84.
        latitude (float): Latitude in WGS84.

    Returns:
        tuple: A tuple containing the x and y RD coordinates.
    """
    transformer_to_rd = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
    x_rd, y_rd = transformer_to_rd.transform(longitude, latitude)
    return x_rd, y_rd


def rd_to_wgs84(x_rd, y_rd):
    """
    Converts RD New (EPSG:28992) coordinates to WGS84 coordinates.

    Args:
        x_rd (float): X coordinate in RD New.
        y_rd (float): Y coordinate in RD New.

    Returns:
        tuple: A tuple containing the longitude and latitude in WGS84.
    """
    transformer_to_wgs84 = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)
    longitude, latitude = transformer_to_wgs84.transform(x_rd, y_rd)
    return longitude, latitude


def get_lon_lat_from_address(address):
    """
    Retrieves the longitude and latitude for a given address using the Nominatim API.

    Args:
        address (str): The address to geocode.

    Returns:
        tuple: A tuple containing the longitude and latitude, or (None, None) if not found.
    """
    # Define the endpoint and parameters for the Nominatim API
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": address, "format": "json", "limit": 1}

    # Include the User-Agent header
    headers = {"User-Agent": "BGTFetcher/1.0 (test@test.com)"}

    # Make a GET request to the API with headers
    response = requests.get(url, params=params, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        if data:
            lat = float(data[0]["lat"])
            lon = float(data[0]["lon"])
            return lon, lat
        else:
            print("No results found for the address.")
            return None, None
    else:
        print(f"Error fetching coordinates for address: {response.status_code}")
        return None, None


def get_additional_address_info(postcode, huisnummer):
    """Use the provided postcode and huisnummer to obtain straatnaam from the bag api"""
    url = cfg.ENDPOINT_BAG
    params = {
        "postcode": postcode,
        "huisnummer": huisnummer,
    }
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        if data:
            if data["_embedded"]["bagadresinformatie"][0]["openbareruimteNaam"]:
                return data["_embedded"]["bagadresinformatie"][0]["openbareruimteNaam"]
    else:
        return ""
