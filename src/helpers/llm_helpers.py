"""
Module for handling of LLMs and prompting them.
Currently supports the OpenAI models on Azure
as well as some HuggingFace models.
"""
import logging
import os
import sys
sys.path.append("..")

from abc import ABC, abstractmethod

import torch
from .llm_config import MODEL_MAPPING
from openai import AzureOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

from pydantic import Field
from typing import Any, Dict, Iterator, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
# from langchain.chat_models import ChatOpenAI, AzureChatOpenAI


class UnsupportedModelError(Exception):
    """Exception raised for unsupported models."""

    pass


# class StoppingCriteriaSub(StoppingCriteria):

#     def __init__(self, stops = [], encounters=1):
#       super().__init__()
#       self.stops = stops
#       self.ENCOUNTERS = encounters

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
#       stop_count = 0
#       for stop in self.stops:
#         stop_count = (stop == input_ids[0]).sum().item()

#       if stop_count >= self.ENCOUNTERS:
#           return True
#       return False


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


class CustomLLM(LLM):
    """Base LLM class"""

    @abstractmethod
    def prompt(self, prompt, context=None, system=None, response_format=None):
        """Function to prompt model should always be implemented"""
        raise NotImplementedError("Implement prompt function")


class HuggingFaceLLM(CustomLLM):
    """A class to handle self-hosted HG models"""
    model_name = ""
    hf_token = ""
    hf_cache = ""
    params = {}
    model:Any = None
    tokenizer:Any = None

    def _load_model(self):
        """
        Load HF model based on a short model name.
        Expects known mapping to full model ID & params
        """
        logging.info(f"Loading {self.model_name}")
        if self.model_name not in MODEL_MAPPING:
            raise UnsupportedModelError(f"Unsupported Model {self.model_name}. Choose from {MODEL_MAPPING.keys()}")

        model_config = MODEL_MAPPING[self.model_name]
        model_id = model_config["id"]
        kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "token": self.hf_token,
        }
        kwargs.update(model_config["kwargs"])

        self.model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=self.hf_cache, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=self.hf_cache, **kwargs)


    def get_stopping_criteria(self, stop):
        if not self.model:
            self._load_model()

        stop_words_ids = [
            self.tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze() 
            for stop_word in stop
        ]

        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        return stopping_criteria


    def prompt(self, prompt, stop=None, context=None, system=None, force_format=None):

        if not self.model:
            self._load_model()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
        attention_mask = torch.ones(input_ids.shape).to(device)

        if stop:
            stopping_criteria = self.get_stopping_criteria(stop)
        else:
            stopping_criteria = None

        output = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
            **self.params,
        )

        response = self.tokenizer.decode(output[0], skip_special_tokens=True).removeprefix(prompt)
        # print(response)

        return response


    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "CustomHuggingFaceModel"


    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        """Run the LLM on the given input.
        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            The model output as a string. Actual completions SHOULD NOT include the prompt.
        """

        return self.prompt(prompt, stop=stop)


class OpenAILLM(CustomLLM):
    """
    A class to support use of OpenAI LLMs.
    Expects Azure deployment and corresponding endpoint, key, etc.
    """
    model_name = ""
    api_endpoint = ""
    api_key = ""
    api_version = ""
    params = {}
    client:Any = None

    def _get_client(self):
        client = AzureOpenAI(
            azure_endpoint=self.api_endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
        )

        return client


    def prompt(self, prompt, stop=None, context=None, system=None, force_format=None):
        """
        Prompt model.
        Starts from empty history.
        #TODO: pass full chat history.
        """

        if not self.client:
            self.client = self._get_client()

        conversation = []

        if system:
            conversation.append({"role": "system", "content": system})

        if context:
            conversation.append({"role": "system", "content": context})

        conversation.append({"role": "user", "content": prompt})

        if force_format:
            if force_format == "json":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=conversation,
                    **self.params,
                    stop=stop,
                    response_format={"type": "json_object"},
                )

            else:
                raise NotImplementedError(
                    f"Currently there is no support for special formats other than json"
                )

        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=conversation,
                **self.params,
                stop=stop,
            )

        finish_reason = response.choices[0].finish_reason
        if finish_reason != "stop":
            print(finish_reason)

        return response.choices[0].message.content


    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "CustomGPTDeployment"


    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        """Run the LLM on the given input.
        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            The model output as a string. Actual completions SHOULD NOT include the prompt.
        """

        return self.prompt(prompt, stop=stop)


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
        hf_cache=None,
        params=None,
    ):
        """
        Get corresponding model.
        #TODO: Add support for HF deployments on Azure
        """
        logging.info(f"Getting a model. Provider: {provider}; Model: {model_name}")

        if provider == "azure":
            if "gpt" in model_name:
                return OpenAILLM(
                    model_name=model_name,
                    api_endpoint=api_endpoint,
                    api_key=api_key,
                    api_version=api_version,
                    params=params)
            else:
                raise NotImplementedError(
                    "Currently there is no support for models other than GPT on Azure."
                )
        elif provider == "huggingface":
            return HuggingFaceLLM(
                model_name=model_name,
                hf_token=hf_token,
                hf_cache=hf_cache,
                params=params)
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

    model = LLMRouter.get_model(
        provider="huggingface",
        model_name="mistral-7b-instruct",
        hf_token=os.environ["HF_TOKEN"],
        hf_cache=os.environ["HF_CACHE"],
        params=hf_params,
    )
    print(f"HF Response to {test}!: {model.prompt(test)}")
