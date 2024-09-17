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
from helpers.waste_rag_helpers import (
    WasteCollectionInfo
)

import config as cfg

class AfvalProcessor:
    """
    A class to specifically handle 'Afval' type meldingen by extending the base MeldingProcessor
    and implementing specific logic to resolve 'Afval' complaints.
    """

    def __init__(self, melding, model_name, base64_image=None, chat_history=None, melding_attributes=None):
        self.melding = melding
        self.model_name = model_name
        self.base64_image = base64_image
        self.chat_history = chat_history or []
        self.melding_attributes = melding_attributes or {}

    def process_afval(self):
        """
        Process the 'Afval' melding by following specific steps to handle and resolve
        'Afval' complaints.
        """
        logging.info("Processing 'Afval' melding...")
        response = []

        # Step 1: Check if image is uploaded
        if self.base64_image and not self.melding_attributes.get('IMAGE_CAPTION'):
            self.melding_attributes['IMAGE_CAPTION'] = generate_image_caption(self.base64_image)

        # Step 2: Create and send the initial response if not done already
        if not self.melding_attributes.get('INITIAL_RESPONSE'):
            self.generate_initial_response()  # Generate the initial response
            response.append(self.melding_attributes['INITIAL_RESPONSE'])
            if not self.melding_attributes.get('IMAGE_CAPTION'):
                response.append("Je kan gedurende dit proces ook altijd een foto van je melding \
                                toevoegen via de knop onderaan de pagina.")
                    
        # Thank for image upload if image is uploaded
        elif self.melding_attributes.get('INITIAL_RESPONSE') and self.base64_image:
            response.append('Dankjewel voor de foto.')

        # Step 2.5: Determine subtype of 'Afval'
        if not self.melding_attributes.get('SUBTYPE'):
            # Try to get the subtype from the user's messages
            self.melding_attributes.update(get_melding_attributes(self.melding, 'SUBTYPE', self.model_name, self.chat_history))
            # If still not found, ask the user
            if not self.melding_attributes.get('SUBTYPE'):
                subtype_prompt = self._build_subtype_prompt()
                response.append(subtype_prompt)
                add_chat_response(self.chat_history, self.melding, response)
                return

        # Step 3: Obtain address details from chat history
        if not self.melding_attributes.get('ADDRESS'):
            self.melding_attributes.update(get_melding_attributes(self.melding, 'ADDRESS', self.model_name, self.chat_history))

        # Create prompt if address attributes (e.g., postcode) are missing
        address_prompt = self._build_address_prompt()

        # Ask user for address attributes if missing, otherwise create the address attribute
        if address_prompt:
            response.append(address_prompt)
            add_chat_response(self.chat_history, self.melding, response)
            return
        # If address_prompt is empty, we generate the address attribute from straatnaam, huisnummer and postcode
        else:
            self._generate_address()

        # Step 4: Provide user information to solve melding using RAG
        if not self.melding_attributes.get('RAG_INFO'):
            self._build_rag_information()
            response.append(self.melding_attributes['RAG_INFO'])
            follow_up_message = "Is deze informatie voldoende om je melding voorlopig op te lossen (ja/nee)?"
            response.append(follow_up_message)
            add_chat_response(self.chat_history, self.melding, response)
            return

        # Step 5: Process user response on RAG information
        final_response = self._get_final_response()
        response.append(final_response)
        add_chat_response(self.chat_history, self.melding, response)
        logging.info('Afval melding processing completed.')
        return

    def generate_initial_response(self):
        """
        Generate the initial response specific to 'Afval' meldingen.
        """
        prompt_template = ChatPromptTemplate.from_template(cfg.INITIAL_MELDING_TEMPLATE)
        prompt = prompt_template.format(
            melding=self.melding,
            type=self.melding_attributes['TYPE']
        )

        if cfg.ENDPOINT == 'local':
            client = OpenAI(api_key=cfg.API_KEYS["openai"])

        elif cfg.ENDPOINT == 'azure':
            client = AzureOpenAI(
                azure_endpoint = cfg.ENDPOINT_AZURE, 
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

    def generate_rag_response(self):
        """
        Generate RAG response which is relevant given the 'Afval' melding.
        """
        prompt_template = ChatPromptTemplate.from_template(cfg.RESTAFVAL_COLLECTION_INFO_RAG_TEMPLATE)
        prompt = prompt_template.format(
            waste_info=self.waste_info,
            history = self.chat_history,
            melding=self.melding_attributes['INITIAL_MELDING'],
            subtype = self.melding_attributes['SUBTYPE'],
            date_time=self.date_time
        )

        if cfg.ENDPOINT == 'local':
            client = OpenAI(api_key=cfg.API_KEYS["openai"])

        elif cfg.ENDPOINT == 'azure':
            client = AzureOpenAI(
                azure_endpoint = cfg.ENDPOINT_AZURE, 
                api_key=cfg.API_KEYS["openai_azure"],  
                api_version="2024-02-15-preview"
            )

        completion = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": cfg.SYSTEM_CONTENT_RAG_FOR_WASTE},
                {"role": "user", "content": prompt}
            ]
        )
                
        self.melding_attributes['RAG_INFO'] = completion.choices[0].message.content
            
    def _build_subtype_prompt(self):
        """
        Build a prompt to request the subtype of 'Afval' from the user.

        Returns:
            str: The subtype prompt message.
        """
        return "Welk type afval gaat het om? Kies uit 'restafval' of 'grof afval'."

    def _build_address_prompt(self):
        """
        Build a prompt to request missing address information from the user specific to 'Afval' meldingen.

        Returns:
            str: The address prompt message or None if all required fields are provided.
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
        Generate a complete address for the 'Afval' melding using available attributes.
        """
        self.melding_attributes['ADDRESS'] = {
            'STRAATNAAM': self.melding_attributes.get('STRAATNAAM', ''),
            'HUISNUMMER': self.melding_attributes.get('HUISNUMMER', ''),
            'POSTCODE': self.melding_attributes.get('POSTCODE', '')
        }

    def _build_rag_information(self):
        """
        Build relevant information from the RAG system specific to 'Afval' and ask the user if this resolves their melding.
        If the address is fully available from the first user message, add the initial response, RAG info, and follow-up message in one go.
        """
        address = self.melding_attributes.get('ADDRESS')
        day = datetime.now().strftime('%A')
        current_time = datetime.now()
        time = current_time.strftime("%H:%M")
        self.date_time = f"{day} {time}"
        print(self.date_time)

        collector = WasteCollectionInfo(address['STRAATNAAM'], address['HUISNUMMER'], address['POSTCODE'])
        self.waste_info = collector.get_collection_times()
        print(self.waste_info)
        self.generate_rag_response()

    def _get_final_response(self):
        """
        Handle the user's final response to determine if their 'Afval' issue is resolved.
        """
        response_map = {
            'ja': "Fijn te horen, nog een prettige dag.",
            'nee': "Je kan via hier de melding afmaken: LINK"
        }
        response_message = response_map.get(self.melding.lower(), "Kan je alsjeblieft 'ja' of 'nee' antwoorden?")
        return response_message