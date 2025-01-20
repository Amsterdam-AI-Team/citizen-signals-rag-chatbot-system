import logging
import os
import re
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from openai import AzureOpenAI, OpenAI

import config as cfg
import my_secrets
from helpers.melding_helpers import (
    add_chat_response,
    check_postcode_format,
    generate_image_caption,
    get_additional_address_info,
    get_melding_attributes,
)

# Environment setup
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class MeldingProcessor:
    """
    Base class to process general complaints. Designed to handle generic melding-specific logic,
    including identifying intents and extracting entities.
    """

    def __init__(
        self, melding, model_name, base64_image=None, chat_history=None, melding_attributes=None
    ):
        """
        Initialize the MeldingProcessor with the melding text, model name, and optional attributes.
        """
        self.melding = melding
        self.model_name = model_name
        self.base64_image = base64_image
        self.chat_history = chat_history or []
        self.melding_attributes = melding_attributes or {}

    def process_melding(self):
        """
        Main method to process the melding by requesting user information and providing agentic AI information.
        """
        logging.info("Starting melding processing with attributes: %s", self.melding_attributes)
        response = []

        # Step 1: Check if image is uploaded and generate image caption
        if self.base64_image and not self.melding_attributes.get("IMAGE_CAPTION"):
            self.melding_attributes["IMAGE_CAPTION"] = generate_image_caption(self.base64_image)

            # Attempt to extract license plate from image caption
            if not self.melding_attributes.get("LICENSE_PLATE"):
                extracted_attributes = get_melding_attributes(
                    self.melding_attributes["IMAGE_CAPTION"],
                    "LICENSE_PLATE",
                    self.model_name,
                    self.chat_history,
                )
                if extracted_attributes:
                    self.melding_attributes.update(extracted_attributes)

        # Step 2: Establish that melding is clear and has enough information to process it
        if not self.melding_attributes.get("TYPE"):
            extracted_attributes = get_melding_attributes(
                self.melding, "TYPE", self.model_name, self.chat_history
            )
            if extracted_attributes:
                self.melding_attributes.update(extracted_attributes)
                self.melding_attributes["MELDING"] = self.melding
            else:
                response.append(
                    "Kan je iets uitgebreider aangeven waar je een melding van zou willen maken?"
                )
                add_chat_response(self.chat_history, self.melding, response)
                return

        # Step 3: Generate initial response if not already done
        if not self.melding_attributes.get("INITIAL_RESPONSE"):
            self._generate_initial_response()
            response.append(self.melding_attributes["INITIAL_RESPONSE"])
            if not self.melding_attributes.get("IMAGE_CAPTION"):
                response.append(
                    "Je kan gedurende dit proces ook altijd een foto van je melding toevoegen via de knop onderaan de pagina."
                )
            add_chat_response(self.chat_history, self.melding, response)

        # Thank for image upload if image is uploaded
        elif self.melding_attributes.get("INITIAL_RESPONSE") and self.base64_image:
            response.append("Dankjewel voor de foto.")

        # Step 4: Obtain address details from chat history
        if not self.melding_attributes.get("ADDRESS"):
            extracted_attributes = get_melding_attributes(
                self.melding, "ADDRESS", self.model_name, self.chat_history
            )
            if extracted_attributes:
                self.melding_attributes.update(extracted_attributes)

        # Create prompt if address attributes (e.g., postcode) are missing
        address_prompt = self._build_address_prompt()

        # Ask user for address attributes if missing, otherwise create the address attribute
        if address_prompt:
            response.append(address_prompt)
            add_chat_response(self.chat_history, self.melding, response)
            return
        else:
            self._generate_address()

        # Step 5: Retrieve license plate if necessary
        if self._melding_requires_license_plate():
            if not self.melding_attributes.get("LICENSE_PLATE"):
                # Attempt to extract from melding if not already done
                extracted_attributes = get_melding_attributes(
                    self.melding, "LICENSE_PLATE", self.model_name, self.chat_history
                )
                if extracted_attributes:
                    self.melding_attributes.update(extracted_attributes)
                else:
                    # Prompt the user for the license plate
                    response.append("Kun je me het kenteken van het voertuig geven?")
                    add_chat_response(self.chat_history, self.melding, response)
                    return

        # Step 6: Build and execute agentic AI Plan
        if not self.melding_attributes.get("AGENTIC_INFORMATION"):
            response.append(
                "We gaan nu zoeken in onze interne documenten of we je behulpzame informatie kunnen geven die je melding nu al kan oplossen."
            )

            # Delegate to CentralAgent to build and execute agentic AI plan
            from central_agentic_agent import CentralAgent

            central_agent = CentralAgent(self.melding_attributes, self.chat_history)
            central_agent.build_and_execute_plan()
            self.melding_attributes = central_agent.melding_attributes

            # Share agentic AI information with user
            response.append(self.melding_attributes["AGENTIC_INFORMATION"])
            response.append("Wil je de melding alsnog maken?")
            add_chat_response(self.chat_history, self.melding, response)
            return

        # Step 7: Ask user if they wish to escalate melding or not
        response.append(self._get_final_response())
        add_chat_response(self.chat_history, self.melding, response)
        return

    def _generate_initial_response(self):
        """
        Generate the initial response specific to the melding type.
        """
        prompt_template = ChatPromptTemplate.from_template(cfg.INITIAL_MELDING_TEMPLATE)
        prompt = prompt_template.format(melding=self.melding, type=self.melding_attributes["TYPE"])

        if cfg.ENDPOINT == "local":
            client = OpenAI(api_key=my_secrets.API_KEYS["openai"])
        elif cfg.ENDPOINT == "azure":
            client = AzureOpenAI(
                azure_endpoint=cfg.ENDPOINT_AZURE,
                api_key=my_secrets.API_KEYS["openai_azure"],
                api_version="2024-02-15-preview",
            )

        completion = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": cfg.SYSTEM_CONTENT_INITIAL_RESPONSE},
                {"role": "user", "content": prompt},
            ],
        )

        self.melding_attributes["INITIAL_RESPONSE"] = completion.choices[0].message.content

    def _build_address_prompt(self):
        """
        Build a prompt to request missing address information from the user.
        """
        if self.melding_attributes.get("POSTCODE") and not self.melding_attributes.get(
            "HUISNUMMER"
        ):
            return "Bedankt voor de postcode, wat is het huisnummer?"
        if self.melding_attributes.get("HUISNUMMER") and not self.melding_attributes.get(
            "POSTCODE"
        ):
            return "Bedankt voor het huisnummer, wat is de postcode (in format 9999XX)?"
        if not self.melding_attributes.get("POSTCODE") or not self.melding_attributes.get(
            "HUISNUMMER"
        ):
            return "Kan je me de postcode (in format 9999XX) en het huisnummer geven van het adres waar je het probleem ervaart?"
        return None

    def _generate_address(self):
        """
        Generate a complete address for the melding using available attributes.
        """
        self.melding_attributes["STRAATNAAM"] = get_additional_address_info(self)
        self.melding_attributes["HUISNUMMER"] = re.match(
            r"^\d+", self.melding_attributes.get("HUISNUMMER", "")
        ).group()

        self.melding_attributes["ADDRESS"] = {
            "STRAATNAAM": self.melding_attributes.get("STRAATNAAM", ""),
            "HUISNUMMER": self.melding_attributes.get("HUISNUMMER", ""),
            "POSTCODE": self.melding_attributes.get("POSTCODE", ""),
        }

    def _melding_requires_license_plate(self):
        """
        Use LLM to determine whether the melding requires a license plate.
        """
        # Check if we already determined if license plate is needed
        if "LICENSE_PLATE_NEEDED" in self.melding_attributes:
            return self.melding_attributes["LICENSE_PLATE_NEEDED"]

        prompt = f"Is het voor de volgende melding nodig om het kenteken van een voertuig te hebben om de melding te verwerken? Melding: \
            '{self.melding_attributes['MELDING']}'. Antwoord met 'ja' of 'nee'"

        # Make an LLM call
        if cfg.ENDPOINT == "local":
            client = OpenAI(api_key=my_secrets.API_KEYS["openai"])
        elif cfg.ENDPOINT == "azure":
            client = AzureOpenAI(
                azure_endpoint=cfg.ENDPOINT_AZURE,
                api_key=my_secrets.API_KEYS["openai_azure"],
                api_version="2024-02-15-preview",
            )

        completion = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "Je bent een assistent die bepaalt of een kenteken nodig is om een melding te verwerken.",
                },
                {"role": "user", "content": prompt},
            ],
        )

        response = completion.choices[0].message.content.strip().lower()

        # Parse the response
        if "ja" in response:
            self.melding_attributes["LICENSE_PLATE_NEEDED"] = True
            return True
        else:
            self.melding_attributes["LICENSE_PLATE_NEEDED"] = False
            return False

    def _get_final_response(self):
        """
        Handle the user's final response to determine if their issue is resolved.
        """
        response_map = {
            "ja": "Je kan via hier de melding afmaken: https://meldingen.amsterdam.nl/incident/beschrijf",
            "nee": "Fijn te horen, nog een prettige dag.",
        }
        response_message = response_map.get(
            self.melding.lower(), "Kan je alsjeblieft 'ja' of 'nee' antwoorden?"
        )
        return response_message
