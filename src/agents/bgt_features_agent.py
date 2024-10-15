import requests
from pyproj import Transformer
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import matplotlib.pyplot as plt
import contextily as ctx

class BGTAgent:
    """
    A class to fetch and process BGT (Basisregistratie Grootschalige Topografie) features for a given address.

    Attributes:
        base_url (str): Base URL for the PDOK BGT API.
        transformer_to_rd (Transformer): Transformer to convert WGS84 coordinates to RD New (EPSG:28992).
        transformer_to_wgs84 (Transformer): Transformer to convert RD New coordinates to WGS84 (EPSG:4326).
        address (str): The address provided by the user.
        longitude (float): Longitude of the address.
        latitude (float): Latitude of the address.
    """

    def __init__(self, straatnaam: str, huisnummer: str, postcode: str):
        """
        Initializes the BGTFetcher with the provided address.

        Args:
            address (str): The address to fetch BGT features for.

        Raises:
            ValueError: If coordinates cannot be retrieved for the given address.
        """
        self.base_url = 'https://api.pdok.nl/lv/bgt/ogc/v1'
        self.transformer_to_rd = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
        self.transformer_to_wgs84 = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)
        address = f"{straatnaam} {huisnummer}, {postcode}"
        self.address = address
        # Get longitude and latitude during initialization
        self.longitude, self.latitude = self.get_lat_lon_from_address(address)
        if self.longitude is None or self.latitude is None:
            raise ValueError("Could not retrieve coordinates for the given address.")

    def wgs84_to_rd(self, longitude, latitude):
        """
        Converts WGS84 coordinates to RD New (EPSG:28992) coordinates.

        Args:
            longitude (float): Longitude in WGS84.
            latitude (float): Latitude in WGS84.

        Returns:
            tuple: A tuple containing the x and y RD coordinates.
        """
        x_rd, y_rd = self.transformer_to_rd.transform(longitude, latitude)
        return x_rd, y_rd

    def rd_to_wgs84(self, x_rd, y_rd):
        """
        Converts RD New (EPSG:28992) coordinates to WGS84 coordinates.

        Args:
            x_rd (float): X coordinate in RD New.
            y_rd (float): Y coordinate in RD New.

        Returns:
            tuple: A tuple containing the longitude and latitude in WGS84.
        """
        longitude, latitude = self.transformer_to_wgs84.transform(x_rd, y_rd)
        return longitude, latitude

    def get_lat_lon_from_address(self, address):
        """
        Retrieves the longitude and latitude for a given address using the Nominatim API.

        Args:
            address (str): The address to geocode.

        Returns:
            tuple: A tuple containing the longitude and latitude, or (None, None) if not found.
        """
        # Define the endpoint and parameters for the Nominatim API
        url = 'https://nominatim.openstreetmap.org/search'
        params = {
            'q': address,
            'format': 'json',
            'limit': 1
        }

        # Include the User-Agent header
        headers = {
            'User-Agent': 'BGTFetcher/1.0 (test@test.com)'
        }

        # Make a GET request to the API with headers
        response = requests.get(url, params=params, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            if data:
                lat = float(data[0]['lat'])
                lon = float(data[0]['lon'])
                return lon, lat
            else:
                print("No results found for the address.")
                return None, None
        else:
            print(f"Error fetching coordinates for address: {response.status_code}")
            return None, None

    def get_collections(self):
        """
        Retrieves a list of collection IDs from the PDOK BGT API.

        Returns:
            list: A list of collection IDs available in the BGT API.
        """
        response = requests.get(f'{self.base_url}/collections', params={'f': 'json'})
        if response.status_code == 200:
            data = response.json()
            collections = data.get('collections', [])
            collection_ids = [collection.get('id') for collection in collections]
            return collection_ids
        else:
            print(f'Error getting collections: {response.status_code}')
            return []

    def get_features(self, collection_id, x_rd, y_rd):
        """
        Retrieves features from a specific collection that intersect with a bounding box around the given RD coordinates.

        Args:
            collection_id (str): The collection ID to retrieve features from.
            x_rd (float): X coordinate in RD New.
            y_rd (float): Y coordinate in RD New.

        Returns:
            list: A list of features from the collection within the bounding box.
        """
        delta = 5  # Adjust as needed
        # Create a bounding box in RD coordinates
        bbox_rd = f"{x_rd - delta},{y_rd - delta},{x_rd + delta},{y_rd + delta}"
        # Convert RD bbox to WGS84 bbox
        ll_longitude, ll_latitude = self.rd_to_wgs84(x_rd - delta, y_rd - delta)
        ur_longitude, ur_latitude = self.rd_to_wgs84(x_rd + delta, y_rd + delta)
        bbox_wgs84 = f"{ll_longitude},{ll_latitude},{ur_longitude},{ur_latitude}"

        params = {
            'bbox': bbox_wgs84,
            'f': 'json',
            'limit': 1000,
        }
        url = f'{self.base_url}/collections/{collection_id}/items'
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            features = data.get('features', [])
            return features
        else:
            print(f'Error getting features for collection {collection_id}: {response.status_code}')
            return []

    def get_bgt_features_at_coordinate(self):
        """
        Retrieves BGT features at the stored coordinate (longitude and latitude).

        Returns:
            GeoDataFrame or None: A GeoDataFrame containing the features that contain the point, or None if no features are found.
        """
        # Use the stored longitude and latitude
        x_rd, y_rd = self.wgs84_to_rd(self.longitude, self.latitude)
        collection_ids = self.get_collections()
        all_features = []

        # Optionally, limit to specific collections to speed up the process
        # collection_ids = ['pand', 'wegdeel', 'waterdeel']

        for collection_id in collection_ids:
            features = self.get_features(collection_id, x_rd, y_rd)
            if features:
                for feature in features:
                    # Add the collection ID to the feature properties
                    feature['properties']['collection_id'] = collection_id
                    all_features.append(feature)

        if all_features:
            # Create a GeoDataFrame from the features
            gdf = gpd.GeoDataFrame.from_features(all_features)
            # Ensure the geometry is correctly set
            if 'geometry' in gdf.columns:
                gdf.set_geometry('geometry', inplace=True)
            else:
                print('No geometry column found in the data.')
                return None

            # Create a Point geometry for the coordinate
            point_geom = Point(self.longitude, self.latitude)
            # Ensure the GeoDataFrame has the correct CRS
            gdf.set_crs(epsg=4326, inplace=True)
            # Select features that contain the point
            gdf_contains_point = gdf[gdf.contains(point_geom)]

            if not gdf_contains_point.empty:
                # Process gdf_contains_point to keep only the latest 'eind_registratie' per 'functie'
                # Determine the column name for 'functie'
                if 'functie' in gdf_contains_point.columns:
                    functie_column = 'functie'
                elif 'type' in gdf_contains_point.columns:
                    functie_column = 'type'
                else:
                    print("'functie' or 'type' column not found in the data.")
                    functie_column = None  # Proceed without filtering

                # Check if 'eind_registratie' is in columns
                if 'eind_registratie' in gdf_contains_point.columns and functie_column is not None:
                    # Convert 'eind_registratie' to datetime
                    gdf_contains_point = gdf_contains_point.copy()
                    gdf_contains_point['eind_registratie'] = pd.to_datetime(
                        gdf_contains_point['eind_registratie'], utc=True, errors='coerce'
                    )
                    # Sort and drop duplicates to keep the latest 'eind_registratie' per 'functie'
                    gdf_contains_point = (
                        gdf_contains_point.sort_values('eind_registratie', ascending=False, na_position='first')
                        .drop_duplicates(subset=[functie_column], keep='first')
                    )
                else:
                    print("'eind_registratie' column not found or 'functie' column not found. Skipping filtering.")

                return gdf_contains_point
            else:
                print('No features contain the specified coordinate.')
                return None
        else:
            print('No BGT features found for this location.')
            return None

    def plot_features(self, gdf):
        """
        Plots the provided GeoDataFrame and the stored coordinate on a map with a basemap.

        Args:
            gdf (GeoDataFrame): The GeoDataFrame to plot.
        """
        # Reproject the GeoDataFrame to Web Mercator (EPSG:3857) for plotting with contextily
        gdf = gdf.to_crs(epsg=3857)

        # Create a GeoDataFrame for the point
        point_geom = Point(self.longitude, self.latitude)
        point_gdf = gpd.GeoDataFrame([{'geometry': point_geom}], crs='EPSG:4326')
        point_gdf = point_gdf.to_crs(epsg=3857)

        # Calculate the combined bounds of the features and the point
        total_bounds = gdf.total_bounds
        point_bounds = point_gdf.total_bounds
        minx = min(total_bounds[0], point_bounds[0])
        miny = min(total_bounds[1], point_bounds[1])
        maxx = max(total_bounds[2], point_bounds[2])
        maxy = max(total_bounds[3], point_bounds[3])

        # Create a buffer around the bounds
        x_buffer = (maxx - minx) * 0.5  # Adjust the multiplier to increase/decrease buffer
        y_buffer = (maxy - miny) * 0.5
        minx -= x_buffer
        miny -= y_buffer
        maxx += x_buffer
        maxy += y_buffer

        # Plot the features and the point
        fig, ax = plt.subplots(figsize=(10, 10))
        gdf.plot(ax=ax, alpha=0.5, edgecolor='k')
        point_gdf.plot(ax=ax, color='red', markersize=100)

        # Set the plot extent to the buffered bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

        # Add basemap
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

        # Adjust the plot appearance
        ax.set_axis_off()
        plt.show()

    def get_functie_from_gdf(self, gdf):
        if gdf is not None:
            # Iterate over rows in the GeoDataFrame
            for index, row in gdf.iterrows():
                # Check if 'bag_pnd' is not NaN
                if not pd.isna(row['bag_pnd']):
                    return {'bgt_functie': 'pand'}
                # Check if 'functie' is not NaN
                if not pd.isna(row['functie']):
                    return {'bgt_functie': row['functie']}
        # If no valid 'bag_pnd' or 'bgt_functie' is found, return None
        return None

if __name__ == '__main__':
    # Specify the address
    address = 'Amstel, 10, 1017AA'  # Replace with your desired address

    # Instantiate the BGTAgent class with the address
    fetcher = BGTAgent('Schalk Burgerstraat', '103', '1092KP')

    # Get BGT features at the coordinate
    bgt_functie = fetcher.get_bgt_features_at_coordinate()

    print(bgt_functie)
