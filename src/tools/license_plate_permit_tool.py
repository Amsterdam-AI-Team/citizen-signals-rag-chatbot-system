import sys

sys.path.append("..")

import csv
import os

import config as cfg


class LicensePlatePermitTool:
    def __init__(self, license_plate, license_plate_field="license_plate_no"):
        self.csv_file = os.path.join(cfg.PERMITS_PATH, "202410_dummydata_licenseplates.csv")
        self.license_plate = license_plate
        self.license_plate_field = license_plate_field
        self.permits = self.load_permits()

    def load_permits(self):
        permits = {}
        try:
            with open(self.csv_file, mode="r", newline="") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    license_plate = row.get(self.license_plate_field)
                    if license_plate:
                        permits[license_plate.strip().upper()] = {
                            "permit_type": row.get("permit_type", "").strip(),
                            "permit_valid_start_date": row.get(
                                "permit_valid_start_date", ""
                            ).strip(),
                            "permit_valid_end_date": row.get("permit_valid_end_date", "").strip(),
                            "permit_location": row.get("permit_location", "").strip(),
                        }
            return permits
        except FileNotFoundError:
            print(f"Error: File '{self.csv_file}' not found.")
            return permits

    def has_permit(self):
        """
        Retrieve the permit details for a specific license plate.

        :param license_plate: The license plate to check.
        :return: A formatted string with permit details if the license plate has a permit,
                 otherwise a message indicating no permit exists.
        """
        license_plate = self.license_plate.strip().upper()
        permit = self.permits.get(license_plate)
        if permit:
            return (
                f"License Plate: {license_plate}\n"
                f"Permit Type: {permit['permit_type']}\n"
                f"Permit Valid Start Date: {permit['permit_valid_start_date']}\n"
                f"Permit Valid End Date: {permit['permit_valid_end_date']}\n"
                f"Permit Location: {permit['permit_location']}"
            )
        else:
            return f"No permit exists for license plate '{license_plate}'."


if __name__ == "__main__":
    license_plate = "DC-743-SK"

    # Initialize the retriever
    retriever = LicensePlatePermitTool(license_plate=license_plate)
    # Test case: Check if permits were loaded
    if retriever.permits:
        print("Permits loaded successfully.")
    else:
        print("No permits found or file missing.")

    # Test case: Check if a license plate has a permit
    license_plate_with_permit = "DC-743-SK"  # Replace with an actual permit for testing
    print(retriever.has_permit())
