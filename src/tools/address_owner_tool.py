"""
A module to retrieve the owner of a specific address based on street name and house number.
For privacy and security reasons, it currently uses dummy data.
"""
import csv
import os
import sys

sys.path.append("..")
import config as cfg


class AddressOwnerTool:
    """
    A tool to retrieve the owner of a specific address based on street name and house number.

    Attributes:
        csv_file (str): The path to the CSV file containing address and owner information.
        street (str): The street name of the address to look up.
        number (str): The house number of the address to look up.
        owners (dict): A dictionary mapping (street, number) tuples to owners.
    """

    def __init__(self, street, number):
        """
        Initializes the AddressOwnerTool with a specified street and number.

        Args:
            street (str): The street name of the address to look up.
            number (str): The house number of the address to look up.
        """
        self.csv_file = os.path.join(cfg.ADDRESS_OWNERS_PATH, "202411_dummydata_addressowners.csv")
        self.street = street
        self.number = number
        self.owners = self.load_owners()

    def load_owners(self):
        """
        Loads owner information from a CSV file and stores it in a dictionary.

        Each address is stored as a key-value pair, where the key is a tuple (street, number)
        and the value is the owner. This allows for quick lookups based on the combination of
        street and number.

        Returns:
            dict: A dictionary with (street, number) tuples as keys and owner names as values.
        """
        owners = {}
        try:
            with open(self.csv_file, mode="r", newline="") as file:
                reader = csv.DictReader(file, delimiter=";")
                for row in reader:
                    street = row.get("streetname", "").strip().lower()
                    number = row.get("number", "").strip()
                    owner = row.get("owner", "").strip()
                    if street and number and owner:
                        owners[(street, number)] = owner
            return owners
        except FileNotFoundError:
            print(f"Error: File '{self.csv_file}' not found.")
            return owners

    def get_owner(self):
        """
        Retrieves the owner of the specified address (street and number).

        If the address (street, number) exists in the loaded data, it returns the owner's name.
        Otherwise, it returns a message indicating that no specific owner was found.

        Returns:
            str: A message with the owner's name if found, or "No specific owner found."
        """
        street_key = self.street.strip().lower()
        number_key = str(self.number).strip()
        owner = self.owners.get((street_key, number_key))
        if owner:
            return f"Owner address: {owner}"
        else:
            return "No specific owner found."


if __name__ == "__main__":
    # Example usage:
    tool = AddressOwnerTool(street="Stationsplein", number="5")
    print(tool.get_owner())  # Should return the owner if a match is found
