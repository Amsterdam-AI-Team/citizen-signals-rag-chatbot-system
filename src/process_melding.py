import os
import logging

from helpers.melding_helpers import (
    get_melding_attributes, 
    generate_image_caption,
    add_chat_response
)

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
        Main method to process the melding by determining the melding type and delegating to the appropriate handler.
        """
        logging.info("Starting melding processing with attributes: %s", self.melding_attributes)

        # Step 1: Check if image is uploaded
        if self.base64_image and not self.melding_attributes.get('IMAGE_CAPTION'):
            self.melding_attributes['IMAGE_CAPTION'] = generate_image_caption(self.base64_image)

        if not self.melding_attributes.get('TYPE'):
            extracted_attributes = get_melding_attributes(self.melding, 'TYPE', self.model_name, self.chat_history)
            if extracted_attributes:
                self.melding_attributes.update(extracted_attributes)
            else:
                add_chat_response(self.chat_history, self.melding, ["Kan je iets uitgebreider aangeven waar je een melding van zou willen maken?"])
                return

        # Delegate melding to correct processor given type
        if self.melding_attributes['TYPE'] == 'Zwerfvuil':
            from processors import ZwervuilProcessor
            processor = ZwervuilProcessor(self.melding, self.model_name, self.base64_image, self.chat_history, self.melding_attributes)
            processor.process_zwervuil()
            self.chat_history = processor.chat_history
            self.melding_attributes = processor.melding_attributes
        else:
            add_chat_response(self.chat_history, self.melding, ["Dit type melding wordt momenteel niet ondersteund."])
            logging.info('Unsupported melding type: %s', self.melding_attributes['TYPE'])