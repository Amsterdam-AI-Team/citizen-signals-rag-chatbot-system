import logging
from functools import partial
from typing import Optional

import config as cfg
from agents.bgt_features_agent import BGTAgent
from agents.waste_collection_agent import WasteCollectionAgent
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import LLMChain
from langchain.llms import OpenAI, AzureOpenAI
from langchain.memory import ConversationBufferMemory

# Configure logging
# logging.basicConfig(level=logging.INFO)

class CentralAgent:
    """
    An improved agent that builds and executes AI-driven plans to retrieve useful information and generate 
    tailored responses to resolve a 'melding' (incident or issue) before escalating it into the system.
    """

    def __init__(
        self,
        melding_attributes: Optional[dict] = None,
        chat_history: Optional[list] = None,
    ):
        """
        Initialize the CentralAgent with the required attributes to handle the melding.

        Args:
            melding (dict): The melding (incident) details.
            model_name (str): The name of the AI model to use for generating responses.
            base64_image (str, optional): Base64-encoded image, if provided in the melding.
            chat_history (list, optional): List of previous chat messages with the user.
            melding_attributes (dict, optional): Additional attributes related to the melding.
        """
        self.melding_attributes = melding_attributes or {}
        self.chat_history = chat_history or []
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Initialize the language model
        self.llm = self.initialize_llm()

        # Define the tools available to the agent
        self.tools = self.initialize_tools()

        # Initialize the agent
        self.agent_executor = self.initialize_agent_executor()

    def initialize_llm(self):
        """Initialize the language model based on the configuration."""
        if cfg.ENDPOINT == 'local':
            llm = OpenAI(api_key=cfg.API_KEYS["openai"], temperature=0)
        elif cfg.ENDPOINT == 'azure':
            llm = AzureOpenAI(
                azure_endpoint=cfg.ENDPOINT_AZURE,
                api_key=cfg.API_KEYS["openai_azure"],
                api_version="2024-02-15-preview",
                temperature=0,
            )
        else:
            llm = OpenAI(api_key=cfg.API_KEYS["openai"], temperature=0)
        return llm

    def initialize_tools(self):
        """Initialize the tools available to the agent."""
        tools = [
            Tool(
                name="GetWasteCollectionInfo",
                func=partial(self.get_waste_collection_info),
                description=(
                    "Use this tool to get waste collection information given an address "
                    "in the format 'STRAATNAAM, HUISNUMMER, POSTCODE'. Returns 'No information found' if unsuccessful."
                ),
            ),
            Tool(
                name="GetBGTInfo",
                func=partial(self.get_bgt_info),
                description=(
                    "Use this tool to get the BGT function of a given address "
                    "in the format 'STRAATNAAM, HUISNUMMER, POSTCODE'. Returns 'No information found' if unsuccessful."
                ),
            ),
        ]
        return tools

    def initialize_agent_executor(self):
        """Initialize the agent executor with the specified tools and LLM."""
        prompt = self.create_custom_prompt()
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        agent = ZeroShotAgent(llm_chain=llm_chain, tools=self.tools, allowed_tools=[tool.name for tool in self.tools])
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
        )
        return agent_executor

    def create_custom_prompt(self):
        """Create a custom prompt template for the agent using ZeroShotAgent.create_prompt."""

        # Create the prompt using ZeroShotAgent's create_prompt method
        prompt = ZeroShotAgent.create_prompt(
            tools=self.tools,
            prefix=cfg.AGENTIC_AI_AGENT_PROMPT_PREFIX,
            suffix=cfg.AGENTIC_AI_AGENT_PROMPT_SUFFIX,
            format_instructions=cfg.AGENTIC_AI_AGENT_PROMPT_FORMAT_INSTRUCTIONS,
        )
        return prompt

    def build_and_execute_plan(self):
        """
        Build and execute an AI-driven plan to resolve the melding.

        This method orchestrates the plan-building process and then executes the plan to address the melding.
        """
        logging.info("Building and executing plan to solve melding...")

        # Prepare the input prompt for the agent
        melding_text = self.melding_attributes.get('MELDING', '')

        # Process chat_history to create a string representation
        chat_history_entries = []
        for entry in self.chat_history:
            vraag = entry.get('vraag', '')
            antwoord = entry.get('antwoord', '')

            # If 'antwoord' is a list, join it into a single string
            if isinstance(antwoord, list):
                antwoord = "\n".join(antwoord)

            # Append formatted conversation entry
            chat_history_entries.append(f"User: {vraag}\nAssistant: {antwoord}")

        # Join all conversation entries into a single string
        chat_history = "\n".join(chat_history_entries)

        # Prepare inputs for the agent
        inputs = {
            "melding": melding_text,
            "chat_history": chat_history,
        }

        # Run the agent with the input prompt
        try:
            response = self.agent_executor.run(inputs)
            # Store the response in melding_attributes
            self.melding_attributes['AGENTIC_INFORMATION'] = response
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            # Handle the failure accordingly
            self.melding_attributes['AGENTIC_INFORMATION'] = (
                "No useful information was found. Your melding will be escalated in the system."
            )

    def get_waste_collection_info(self, address: str) -> str:
        """
        Retrieve waste collection information based on the provided address.

        Args:
            address (str): The address in the format 'STRAATNAAM, HUISNUMMER, POSTCODE'.

        Returns:
            str: Waste collection information or 'No information found' if unsuccessful.
        """
        logging.info(f"Retrieving waste collection info for address: {address}")
        try:
            straatnaam = self.melding_attributes['STRAATNAAM']
            huisnummer = self.melding_attributes['HUISNUMMER']
            postcode = self.melding_attributes['POSTCODE']
            waste_collection_agent = WasteCollectionAgent(straatnaam, huisnummer, postcode)
            waste_collection_info = waste_collection_agent.get_collection_times()
            if not waste_collection_info:
                return "No information found"
            return waste_collection_info
        except Exception as e:
            logging.error(f"Failed to get waste collection info: {e}")
            return "No information found"

    def get_bgt_info(self, address: str) -> str:
        """
        Retrieve BGT (Basisregistratie Grootschalige Topografie) information based on the provided address.

        Args:
            address (str): The address in the format 'STRAATNAAM, HUISNUMMER, POSTCODE'.

        Returns:
            str: BGT information or 'No information found' if unsuccessful.
        """
        logging.info(f"Retrieving BGT info for address: {address}")
        try:
            straatnaam = self.melding_attributes['STRAATNAAM']
            huisnummer = self.melding_attributes['HUISNUMMER']
            postcode = self.melding_attributes['POSTCODE']
            bgt_info_agent = BGTAgent(straatnaam, huisnummer, postcode)
            gdf_bgt_info = bgt_info_agent.get_bgt_features_at_coordinate()
            if gdf_bgt_info is not None and not gdf_bgt_info.empty:
                return bgt_info_agent.get_functie_from_gdf(gdf_bgt_info)
            else:
                return "No information found"
        except Exception as e:
            logging.error(f"Failed to get BGT info: {e}")
            return "No information found"


# Example usage:
if __name__ == "__main__":
    melding_attributes = {
        "MELDING": "There is grofvuil next to a container in front of my house.",
        "STRAATNAAM": "Keizersgracht",
        "HUISNUMMER": "75",
        "POSTCODE": "1015CE"
    }

    agent = CentralAgent(
        chat_history=[],
        melding_attributes=melding_attributes,
    )
    agent.build_and_execute_plan()