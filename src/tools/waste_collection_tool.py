import requests
from pyproj import Transformer


class WasteCollectionTool:
    """
    A class to fetch and process waste collection information for a given address in Amsterdam.

    Attributes:
        straatnaam (str): Street name of the address.
        huisnummer (str): House number of the address.
        postcode (str): Postal code of the address.
        api_data (dict): Data retrieved from the Amsterdam waste collection API.
        transformer (Transformer): Coordinate transformer to convert from RD to WGS84.
    """

    def __init__(self, straatnaam: str, huisnummer: str, postcode: str):
        """
        Initializes WasteCollectionInfo with the given address and fetches the related waste collection data.

        Args:
            straatnaam (str): The street name of the address.
            huisnummer (str): The house number of the address.
            postcode (str): The postal code of the address.
        """
        self.straatnaam = straatnaam
        self.huisnummer = huisnummer
        self.postcode = postcode
        self.api_data = self._fetch_api_data()
        self.transformer = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)

        print(f"Informatie {self.straatnaam} {self.huisnummer}, {self.postcode}\n")

    def _fetch_api_data(self) -> dict:
        """
        Fetches waste collection data from the Amsterdam API for the provided address.

        Returns:
            dict: The API response containing waste collection information.
        """
        url = f"https://api.data.amsterdam.nl/v1/afvalwijzer/afvalwijzer/?straatnaam={self.straatnaam}&huisnummer={self.huisnummer}"
        response = requests.get(url)
        response.raise_for_status()  # Raises an error if the request failed
        return response.json()

    def _reconstruct_url(
        self,
        rd_x: float,
        rd_y: float,
        pk_vid: str = "72ac279c8dc526301726144520a65bd0",
        layer_code: str = "12493",
    ) -> str:
        """
        Reconstructs a URL to display nearby waste containers on the map using given RD coordinates.

        Args:
            rd_x (float): RD coordinate x (eastings).
            rd_y (float): RD coordinate y (northings).
            pk_vid (str): Unique identifier for session tracking (default is provided).
            layer_code (str): Layer code for the type of waste containers.

        Returns:
            str: A URL pointing to the map showing the location of waste containers.
        """
        # Convert RD coordinates to WGS84 latitude and longitude
        lon, lat = self.transformer.transform(rd_x, rd_y)
        link_template = "https://kaart.amsterdam.nl/afvalcontainers?pk_vid={pk_vid}#17/{lat}/{lon}/brt/{layer_code}///{lat},{lon}"
        return link_template.format(pk_vid=pk_vid, lat=lat, lon=lon, layer_code=layer_code)

    def _process_waste_info(self) -> str:
        """
        Processes and formats the waste collection data based on predefined categories and presentation order.

        Returns:
            str: A formatted string containing the waste collection information for the address.
        """
        afvalwijzer_list = self.api_data["_embedded"]["afvalwijzer"]
        result = []

        # Keep track of the fraction types we have already processed
        processed_fractions = set()

        # Define the desired order of fractions
        fraction_order = ["Rest", "Grof", "Papier", "Glas", "Textiel"]

        # Create a mapping from fraction names to their respective info blocks
        fraction_info_mapping = {fraction: None for fraction in fraction_order}

        # Mapping of fraction names to their respective layer codes and pk_vids for the URLs
        fraction_layer_info = {
            "Rest": {"layer_code": "12496", "pk_vid": "72ac279c8dc526301726144520a65bd0"},
            "Grof": {"layer_code": "12497", "pk_vid": "72ac279c8dc526301726237436a65bd0"},
            "Papier": {"layer_code": "12493", "pk_vid": "72ac279c8dc526301726144520a65bd0"},
            "Glas": {"layer_code": "12492", "pk_vid": "72ac279c8dc526301726144447a65bd0"},
            "Textiel": {"layer_code": "12495,13698", "pk_vid": "72ac279c8dc526301726144470a65bd0"},
        }

        # The Ophaaldag value to omit if present
        ophaaldag_to_omit = "maandag, dinsdag, woensdag, donderdag, vrijdag, zaterdag"

        # Process all items and store them in fraction_info_mapping
        for afval in afvalwijzer_list:
            fraction_name = afval.get("afvalwijzerFractieNaam", "")
            coordinates = afval.get("afvalwijzerGeometrie", {}).get("coordinates", [])

            if fraction_name in processed_fractions:
                continue  # Skip if this fraction is already processed

            if not coordinates:
                continue  # Skip if coordinates are missing

            rd_x, rd_y = coordinates  # RD coordinates are [x, y]

            info_lines = [fraction_name if fraction_name != "Papier" else "Papier en karton"]
            hoe_parts = []

            # Handle afvalwijzerButtontekst and afvalwijzerUrl
            button_text = afval.get("afvalwijzerButtontekst")
            button_url = afval.get("afvalwijzerUrl")
            if button_text and button_url:
                # For 'Grof' fraction, we might need to construct the URL with extra parameters
                if fraction_name == "Grof":
                    # Add the address parameters to the URL if needed
                    postcode = afval.get("postcode")
                    huisnummer = afval.get("huisnummer")
                    huisletter = afval.get("huisletter") or ""
                    huisnummertoevoeging = afval.get("huisnummertoevoeging") or ""
                    address_param = f"{postcode},{huisnummer},{huisletter},{huisnummertoevoeging}"
                    button_url += f"?GUID={address_param}&pk_vid={fraction_layer_info[fraction_name]['pk_vid']}"
                hoe_parts.append(f'<a href="{button_url}">{button_text}</a>')

            # Include afvalwijzerInstructie2
            instructie2 = afval.get("afvalwijzerInstructie2")
            if instructie2:
                # Combine with existing 'Hoe' information
                if hoe_parts:
                    hoe_parts.append(instructie2)
                else:
                    hoe_parts.append(instructie2)

            if hoe_parts:
                hoe = " ".join(hoe_parts)
                info_lines.append(f"Hoe: {hoe}")

            # Retrieve Ophaaldag
            ophaaldag = afval.get("afvalwijzerOphaaldagen2") or afval.get("afvalwijzerOphaaldagen")

            # Exclude Ophaaldag if it matches the specific value to omit
            if ophaaldag and ophaaldag != ophaaldag_to_omit:
                info_lines.append(f"Ophaaldag: {ophaaldag}")

            buit = afval.get("afvalwijzerBuitenzetten")
            if buit:
                info_lines.append(f"Buitenzetten: {buit}")

            waar = afval.get("afvalwijzerWaar")
            if waar:
                if "Kaart met containers in de buurt" in waar:
                    # Reconstruct the URL with appropriate layer code and pk_vid
                    layer_info = fraction_layer_info.get(fraction_name, {})
                    layer_code = layer_info.get("layer_code", "")
                    pk_vid = layer_info.get("pk_vid", "")
                    link = self._reconstruct_url(rd_x, rd_y, pk_vid=pk_vid, layer_code=layer_code)
                    waar = f'<a href="{link}">{waar}</a>'
                info_lines.append(f"Waar: {waar}")

            # Include afvalwijzerAfvalkalenderOpmerking
            remark = afval.get("afvalwijzerAfvalkalenderOpmerking")
            if remark:
                info_lines.append(remark)

            info = "\n".join(info_lines)
            fraction_info_mapping[fraction_name] = info
            processed_fractions.add(fraction_name)

        # Append the information in the desired order
        for fraction in fraction_order:
            if fraction_info_mapping.get(fraction):
                # For "Grof", rename to "Grof afval"
                if fraction == "Grof":
                    fraction_info_mapping[fraction] = fraction_info_mapping[fraction].replace(
                        "Grof", "Grof afval", 1
                    )
                # Replace "Rest" with "Restafval" for consistency
                if fraction == "Rest":
                    fraction_info_mapping[fraction] = fraction_info_mapping[fraction].replace(
                        "Rest", "Restafval", 1
                    )
                result.append(fraction_info_mapping[fraction])

        return "\n\n".join(result)

    def get_collection_times(self) -> str:
        """
        Returns the waste collection information for the given address.

        Returns:
            str: A formatted string with collection times for each waste category.
        """
        return self._process_waste_info()


if __name__ == "__main__":
    # Specify the address
    address = "Amstel, 10, 1017AA"  # Replace with your desired address

    # Instantiate the WasteCollectionTool class with the address
    fetcher = WasteCollectionTool("Schalk Burgerstraat", "103", "1092KP")

    # Get waste collection info at specified address
    waste_collection_info = fetcher.get_collection_times()
    print(waste_collection_info)
