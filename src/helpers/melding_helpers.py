"""
A module containing helpers for processing a report, such as e.g.
generating a caption, handling the sessions, etc
"""
import json
import logging
import os
import re

from langchain_core.prompts import ChatPromptTemplate
from openai import AzureOpenAI, OpenAI

import config as cfg
import my_secrets


def get_melding_attributes(melding, attribute, model_name, chat_history):
    """
    Extract specified melding attribute (e.g., TYPE, ADDRESS) using a prompt-based approach.

    Args:
        attribute (str): The name of the attribute to extract.
    """
    prompt_template = select_prompt_template(attribute)
    if not prompt_template:
        logging.error("No template found for attribute: %s", attribute)
        return {}

    prompt = prompt_template.format(
        melding=melding, history=get_formatted_chat_history(chat_history)
    )

    if cfg.ENDPOINT == "local":
        client = OpenAI(api_key=my_secrets.API_KEYS["openai"])

    elif cfg.ENDPOINT == "azure":
        client = AzureOpenAI(
            azure_endpoint=cfg.ENDPOINT_AZURE,
            api_key=my_secrets.API_KEYS["openai_azure"],
            api_version="2024-02-15-preview",
        )

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": cfg.SYSTEM_CONTENT_ATTRIBUTE_EXTRACTION},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )

    # Load and update attributes based on response
    response_data = json.loads(completion.choices[0].message.content)
    return response_data if response_data else {}


def select_prompt_template(attribute):
    """
    Select the appropriate prompt template based on the specified attribute.

    Args:
        attribute (str): The attribute for which the template is required.

    Returns:
        ChatPromptTemplate: The corresponding prompt template.
    """
    templates = {
        "TYPE": cfg.MELDING_TYPE_TEMPLATE,
        "ADDRESS": cfg.MELDING_ADDRESS_TEMPLATE,
        "LICENSE_PLATE": cfg.LICENSE_PLATE_TEMPLATE,
    }
    return ChatPromptTemplate.from_template(templates.get(attribute))


def generate_image_caption(base64_image):
    """
    Generate discription of image.

    Args:
        base64_image (base64): The image as base64.

    Returns:
        json: A json containing:
            - content (str): Description of image.
            - total_tokens (int): total tokens used to process image.
    """
    if cfg.ENDPOINT == "local":
        client = OpenAI(api_key=my_secrets.API_KEYS["openai"])

    elif cfg.ENDPOINT == "azure":
        client = AzureOpenAI(
            azure_endpoint=cfg.ENDPOINT_AZURE,
            api_key=my_secrets.API_KEYS["openai_azure"],
            api_version="2024-02-15-preview",
        )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Wat staat er op deze afbeelding? \
                                Als er een duidelijk kenteken zichtbaar is, geef deze ook terug in je output.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "low",
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
    )
    return response.choices[0].message.content


def get_formatted_chat_history(chat_history):
    """
    Format the chat history into a string suitable for inclusion in prompts.

    Returns:
        str: The formatted chat history.
    """
    return "\n".join(
        [f"Vraag: {entry['vraag']}\nAntwoord: {entry['antwoord']}" for entry in chat_history]
    )


def check_postcode_format(input_string):
    """
    Check if the provided postcode format is valid.

    Args:
        input_string (str): The postcode string to validate.

    Returns:
        bool: True if the postcode format is valid, False otherwise.
    """
    cleaned_string = input_string.replace(" ", "")
    return bool(re.match(r"^\d{4}[A-Z]{2}$", cleaned_string))


def add_chat_response(chat_history, melding, responses):
    """
    Add a response to the chat history. If the last entry has the same 'vraag',
    append the responses to the 'antwoord' list. Otherwise, create a new entry.

    Args:
        chat_history (list): The chat history list.
        melding (str): The melding question or statement.
        responses (list): The list of response messages to add.
    """
    chat_history.append({"vraag": melding, "antwoord": responses})
    logging.info("Chat history updated: %s", chat_history)


def load_session(path):
    """
    Load session data from the specified session file if it exists.

    Args:
        path (str): The path to the session file.

    Returns:
        list: A list containing the chat history loaded from the session file.
              If the file does not exist, an empty list is returned.
    """
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f).get("chat_history", [])
    return []


def save_session(file, path):
    """
    Save the current chat history to the specified session file.

    Args:
        file (dict or list): The chat history or other data to be saved.
        path (str): The path to the session file.
    """
    with open(path, "w") as f:
        if "session" in path:
            json.dump({"chat_history": file}, f)
        else:
            json.dump(file, f)
