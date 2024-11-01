"""
Module for handling of LLMs and prompting them.
Currently supports the OpenAI models on Azure
as well as some HuggingFace models.
"""
import logging
import os
from abc import ABC, abstractmethod

import torch
from llm_config import MODEL_MAPPING
from openai import AzureOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer


class UnsupportedModelError(Exception):
    """Exception raised for unsupported models."""

    pass


class LLM(ABC):
    """Base LLM class"""

    @abstractmethod
    def prompt(self, prompt, context=None, system=None):
        """Function to prompt model should always be implemented"""
        raise NotImplementedError("Implement prompt function")


class HuggingFaceLLM(LLM):
    """A class to handle self-hosted HG models"""

    def __init__(self, model_name, hf_token, params=None):
        self.model_name = model_name
        self.hf_token = hf_token
        self.params = params or {}
        self._load_model()

    def _load_model(self):
        """
        Load HF model based on a short model name.
        Expects known mapping to full model ID & params
        """
        logging.info(f"Loading {self.model_name}")
        if self.model_name not in MODEL_MAPPING:
            raise UnsupportedModelError(f"Unsupported Model. Choose from {MODEL_MAPPING.keys()}")

        model_config = MODEL_MAPPING[self.model_name]
        model_id = model_config["id"]
        kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "token": self.hf_token,
        }
        kwargs.update(model_config["kwargs"])

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)

    def prompt(self, prompt, context=None, system=None):
        """Promp model. Use cuda if avalable."""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
        attention_mask = torch.ones(input_ids.shape).to(device)

        output = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            **self.params,
        )

        response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return response


class OpenAILLM:
    """
    A class to support use of OpenAI LLMs.
    Expects Azure deployment and corresponding endpoint, key, etc.
    """

    def __init__(self, model_name, api_endpoint, api_key, api_version, params=None):
        self.model_name = model_name
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.api_version = api_version
        self.params = params or {}
        self.client = self._get_client()

    def _get_client(self):
        client = AzureOpenAI(
            azure_endpoint=self.api_endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
        )
        return client

    def prompt(self, prompt, context=None, system=None):
        """
        Prompt model.
        Starts from empty history.
        #TODO: pass full chat history.
        """
        conversation = []

        if system:
            conversation.append({"role": "system", "content": system})

        if context:
            conversation.append({"role": "system", "content": context})

        conversation.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=conversation,
            **self.params,
        )

        finish_reason = response.choices[0].finish_reason
        if finish_reason != "stop":
            print(finish_reason)

        return response.choices[0].message.content


class LLMRouter:
    """Route LLMs depending on model and desired provider."""

    # #TODO: singleton?

    @staticmethod
    def get_model(
        provider,
        model_name="falcon",
        api_endpoint=None,
        api_key=None,
        api_version=None,
        hf_token=None,
        params=None,
    ):
        """
        Get corresponding model.
        #TODO: Add support for HF deployments on Azure
        """
        if provider == "azure":
            if "gpt" in model_name:
                return OpenAILLM(model_name, api_endpoint, api_key, api_version, params)
            else:
                raise NotImplementedError(
                    "Currently there is no support for models other than GPT on Azure."
                )
        elif provider == "huggingface":
            return HuggingFaceLLM(model_name, hf_token, params)
        else:
            raise ValueError(
                f"Unknown provider specified ({provider})."
                "Current support for azure and huggingface only"
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    gpt_params = {
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stop": None,
    }
    hf_params = {
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.65,
        # "top_k": 25,
        "max_new_tokens": 200,
        "no_repeat_ngram_size": 3,
        "num_return_sequences": 1,
    }

    # test = "Test!"
    test = "Hoe maak ik een melding in Amsterdam?"

    # Test GPT
    model = LLMRouter.get_model(
        provider="azure",
        model_name="gpt-4o",
        api_endpoint=os.environ["API_ENDPOINT"],
        api_key=os.environ["API_KEY"],
        api_version=os.environ["API_VERSION"],
        params=gpt_params,
    )
    print(f"GPT Response to {test}!: {model.prompt(test)}")

    # Test HF Model
    HUGGING_CACHE = "/home/azureuser/cloudfiles/code/hugging_cache"
    os.environ["TRANSFORMERS_CACHE"] = HUGGING_CACHE
    os.environ["HF_HOME"] = HUGGING_CACHE

    model = LLMRouter.get_model(
        provider="huggingface",
        model_name="mistral-7b-instruct",
        hf_token=os.environ["HF_TOKEN"],
        params=hf_params,
    )
    print(f"HF Response to {test}!: {model.prompt(test)}")
