import os
import logging
import requests

from datetime import datetime
from openai import OpenAI, AzureOpenAI
from langchain_core.prompts import ChatPromptTemplate

from helpers.melding_helpers import (
    get_melding_attributes, 
    generate_image_caption,
    add_chat_response
)

import config as cfg

# Environment setup
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MeldingProcessor:
    """
    Base class to process general complaints. Designed to handle generic melding-specific logic,
    including identifying intents and extracting entities.
    """

    def __init__(self, melding, model_name, base64_image=None, chat_history=None, melding_attributes=None):
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
        Main method to process the melding by requesting user information and providing agentic ai information.
        """
        logging.info("Starting melding processing with attributes: %s", self.melding_attributes)
        response = []

        # Step 1: Check if image is uploaded
        if self.base64_image and not self.melding_attributes.get('IMAGE_CAPTION'):
            self.melding_attributes['IMAGE_CAPTION'] = generate_image_caption(self.base64_image)

        # Step 2: Establish that melding is clear and has enough information to process it
        if not self.melding_attributes.get('TYPE'):
            extracted_attributes = get_melding_attributes(self.melding, 'TYPE', self.model_name, self.chat_history)
            if extracted_attributes:
                self.melding_attributes.update(extracted_attributes)
                self.melding_attributes['MELDING'] = self.melding
            else:
                response.append("Kan je iets uitgebreider aangeven waar je een melding van zou willen maken?")
                add_chat_response(self.chat_history, self.melding, response)
                return

        # Step 3: Generate initial response if not already done
        if not self.melding_attributes.get('INITIAL_RESPONSE'):
            self._generate_initial_response()
            response.append(self.melding_attributes['INITIAL_RESPONSE'])
            if not self.melding_attributes.get('IMAGE_CAPTION'):
                response.append("Je kan gedurende dit proces ook altijd een foto van je melding toevoegen via de knop onderaan de pagina.")
            add_chat_response(self.chat_history, self.melding, response)

        # Thank for image upload if image is uploaded
        elif self.melding_attributes.get('INITIAL_RESPONSE') and self.base64_image:
            response.append('Dankjewel voor de foto.')

        # Step 4: Obtain address details from chat history
        if not self.melding_attributes.get('ADDRESS'):
            extracted_attributes = get_melding_attributes(self.melding, 'ADDRESS', self.model_name, self.chat_history)
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

        # Step 5: build and execute agentic AI Plan
        if not self.melding_attributes.get('AGENTIC_INFORMATION'):
            response.append("We gaan nu zoeken in onze interne documenten of we je behulpzame informatie kunnen geven die je melding nu al kan oplossen.")

            # Delegate to CentralProcessor to build and execute agentic AI plan
            from central_agentic_agent import CentralAgent
            print(self.chat_history)
            central_agent = CentralAgent(self.melding_attributes, self.chat_history)
            central_agent.build_and_execute_plan()
            self.melding_attributes = central_agent.melding_attributes

            # Share agentic AI information with user
            response.append(self.melding_attributes['AGENTIC_INFORMATION'])
            response.append('Wil je de melding alsnog maken?')
            add_chat_response(self.chat_history, self.melding, response)
            return

        # Step 6: Ask user if they wish to escalate melding or not
        response.append(self._get_final_response())
        add_chat_response(self.chat_history, self.melding, response)
        return

    def _generate_initial_response(self):
        """
        Generate the initial response specific to the melding type.
        """
        # You can customize this method further if needed
        prompt_template = ChatPromptTemplate.from_template(cfg.INITIAL_MELDING_TEMPLATE)
        prompt = prompt_template.format(
            melding=self.melding,
            type=self.melding_attributes['TYPE']
        )

        if cfg.ENDPOINT == 'local':
            client = OpenAI(api_key=cfg.API_KEYS["openai"])
        elif cfg.ENDPOINT == 'azure':
            client = AzureOpenAI(
                azure_endpoint=cfg.ENDPOINT_AZURE, 
                api_key=cfg.API_KEYS["openai_azure"],  
                api_version="2024-02-15-preview"
            )

        completion = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": cfg.SYSTEM_CONTENT_INITIAL_RESPONSE},
                {"role": "user", "content": prompt}
            ]
        )

        self.melding_attributes['INITIAL_RESPONSE'] = completion.choices[0].message.content

    def _build_address_prompt(self):
        """
        Build a prompt to request missing address information from the user.
        """
        if not self.melding_attributes.get('STRAATNAAM'):
            return "Kan je me de straatnaam geven waar je het probleem ervaart?"
        if not self.melding_attributes.get('HUISNUMMER'):
            return "Wat is het huisnummer?"
        if not self.melding_attributes.get('POSTCODE'):
            return "Wat is de postcode?"
        return None

    def _generate_address(self):
        """
        Generate a complete address for the melding using available attributes.
        """
        self.melding_attributes['ADDRESS'] = {
            'STRAATNAAM': self.melding_attributes.get('STRAATNAAM', ''),
            'HUISNUMMER': self.melding_attributes.get('HUISNUMMER', ''),
            'POSTCODE': self.melding_attributes.get('POSTCODE', '')
        }
    
    def _get_final_response(self):
        """
        Handle the user's final response to determine if their 'Afval' issue is resolved.
        """
        response_map = {
            'ja': "Je kan via hier de melding afmaken: https://meldingen.amsterdam.nl/incident/beschrijf",
            'nee': "Fijn te horen, nog een prettige dag."
        }
        response_message = response_map.get(self.melding.lower(), "Kan je alsjeblieft 'ja' of 'nee' antwoorden?")
        return response_message