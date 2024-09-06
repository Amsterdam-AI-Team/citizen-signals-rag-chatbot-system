import logging

from openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate

from helpers.melding_helpers import (
    get_melding_attributes, 
    generate_image_caption,
    add_chat_response
)
import config as cfg

class ZwervuilProcessor:
    """
    A class to specifically handle 'Zwerfvuil' type meldingen by extending the base MeldingProcessor
    and implementing specific logic to resolve 'Zwerfvuil' complaints.
    """

    def __init__(self, melding, model_name, base64_image=None, chat_history=None, melding_attributes=None):
        self.melding = melding
        self.model_name = model_name
        self.base64_image = base64_image
        self.chat_history = chat_history or []
        self.melding_attributes = melding_attributes or {}

    def process_zwervuil(self):
        """
        Process the 'Zwerfvuil' melding by following specific steps to handle and resolve
        'Zwerfvuil' complaints.
        """
        logging.info("Processing 'Zwerfvuil' melding...")
        response = []
        print(self.melding_attributes)

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
                
        # Thank for image upload
        elif self.melding_attributes.get('INITIAL_RESPONSE') and self.base64_image:
            response.append('Dankjewel voor de foto.')


        # Step 3: Obtain and address details from chat history
        if not self.melding_attributes.get('ADDRESS'):
            self.melding_attributes.update(get_melding_attributes(self.melding, 'ADDRESS', self.model_name, self.chat_history))

        # Create prompt if address attributes (e.g., postcode) are missing
        address_prompt = self._build_address_prompt()

        # Ask user for address attributes if missing, otherwise create the address attribute
        if address_prompt:
            response.append(address_prompt)
            add_chat_response(self.chat_history, self.melding, response)
            return
        # If address_prompt is empty, we generate the address atribute from straatnaam, huisnummer and postcode
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
        logging.info('Zwervuil melding processing completed.')
        return

    def generate_initial_response(self):
        """
        Generate the initial response specific to 'Zwerfvuil' meldingen.
        """
        prompt_template = ChatPromptTemplate.from_template(cfg.INITIAL_MELDING_TEMPLATE)
        prompt = prompt_template.format(
            melding=self.melding,
            type=self.melding_attributes['TYPE']
        )
        
        client = OpenAI(api_key=cfg.API_KEYS["openai"])
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
        Build a prompt to request missing address information from the user specific to 'Zwerfvuil' meldingen.

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
        Generate a complete address for the 'Zwerfvuil' melding using available attributes.
        """
        self.melding_attributes['ADDRESS'] = {
            'STRAATNAAM': self.melding_attributes.get('STRAATNAAM', ''),
            'HUISNUMMER': self.melding_attributes.get('HUISNUMMER', ''),
            'POSTCODE': self.melding_attributes.get('POSTCODE', '')
        }

    def _build_rag_information(self):
        """
        build relevant information from the RAG system specific to 'Zwerfvuil' and ask the user if this resolves their melding.
        If the address is fully available from the first user message, add the initial response, RAG info, and follow-up message in one go.
        """
        address = self.melding_attributes.get('ADDRESS')
        address_string = f"{address['STRAATNAAM']} {address['HUISNUMMER']}, {address['POSTCODE']}"
        self.melding_attributes['RAG_INFO'] = f"De standaard ophaaldagen van afval op adres {address_string} zijn \
            maandag en vrijdag. We verwachten dat het afval daarom morgen is opgehaald."

    def _get_final_response(self):
        """
        Handle the user's final response to determine if their 'Zwerfvuil' issue is resolved.
        """
        response_map = {
            'ja': "Fijn te horen, nog een prettige dag.",
            'nee': "Je kan via hier de melding afmaken: LINK"
        }
        response_message = response_map.get(self.melding.lower(), "Kan je alsjeblieft 'ja' of 'nee' antwoorden?")
        return response_message