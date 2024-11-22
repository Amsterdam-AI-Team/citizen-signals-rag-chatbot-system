import logging
from functools import partial
from typing import Optional
from datetime import datetime
import os

import config as cfg
import my_secrets as sc
from tools.bgt_features_tool import BGTTool
from tools.waste_collection_tool import WasteCollectionTool
from tools.policy_retriever_tool import PolicyRetrieverTool
from tools.license_plate_permit_tool import LicensePlatePermitTool
from tools.meldingen_tool import MeldingenRetrieverTool
from tools.noise_permits_tool import NoisePermitsTool
from tools.address_owner_tool import AddressOwnerTool
from helpers.melding_helpers import get_formatted_chat_history
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from codecarbon import EmissionsTracker

class CentralAgent:
    """
    An improved agent that builds and executes AI-driven plans to retrieve useful information and generate 
    tailored responses to resolve a 'melding' (incident or issue) before escalating it into the system.
    """

    def __init__(
        self,
        llm,
        melding_attributes: Optional[dict] = None,
        chat_history: Optional[list] = None,
    ):
        """
        Initialize the CentralAgent with the required attributes to handle the melding.

        Args:
            melding_attributes (dict, optional): Additional attributes related to the melding.
            chat_history (list, optional): List of previous chat messages with the user.
        """
        self.melding_attributes = melding_attributes or {}
        self.chat_history = chat_history or []
        self.memory = ConversationBufferMemory(memory_key="chat_history", 
                                               input_key="melding", return_messages=True)

        # Initialize the language model
        self.llm = llm

        # Define the tools available to the agent
        self.tools = self.initialize_tools()

        # Initialize the agent
        self.agent_executor = self.initialize_agent_executor()


    def initialize_tools(self):
        """
        Initialize the tools available to the agent.
        """
        melding = self.melding_attributes['MELDING']
        straatnaam = self.melding_attributes['STRAATNAAM']
        huisnummer = self.melding_attributes['HUISNUMMER']
        postcode = self.melding_attributes['POSTCODE']
        not_allowed_tools = []

        os.environ["TRANSFORMERS_CACHE"] = cfg.HUGGING_CACHE
        os.environ["HF_HOME"] = cfg.HUGGING_CACHE

        self.WasteCollectionTool = WasteCollectionTool(straatnaam, huisnummer, postcode)
        self.BGTTool = BGTTool(straatnaam, huisnummer, postcode)
        self.AddressOwnerTool = AddressOwnerTool(straatnaam, huisnummer)
        self.NoisePermitsTool = NoisePermitsTool(straatnaam, huisnummer, postcode, melding)
        self.PolicyRetrieverTool = PolicyRetrieverTool(melding)
        if self.melding_attributes['LICENSE_PLATE_NEEDED'] == True:
            license_plate = self.melding_attributes['LICENSE_PLATE']
            if license_plate:
                self.LicensePlatePermitTool = LicensePlatePermitTool(license_plate)
            else:
                not_allowed_tools.append('GetLicensePlatePermitInfo')
                logging.warning("LICENSE_PLATE_NEEDED is True but LICENSE_PLATE is missing.")
        else:
            not_allowed_tools.append('GetLicensePlatePermitInfo')

        self.MeldingenRetrieverTool = MeldingenRetrieverTool(
            cfg.embedding_model_name, cfg.meldingen_dump, cfg.index_storage_folder)

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
                    "Use this tool to get the BGT function of a given address, in the format 'STRAATNAAM, HUISNUMMER, POSTCODE'."
                    "Returns 'No information found' if unsuccessful."
                ),
            ),
            Tool(
                name="GetAddressOwnerInfo",
                func=partial(self.get_address_owner_info),
                description=(
                    "Use this tool to get the owner of a given address, in the format 'STRAATNAAM, HUISNUMMER'."
                    "The function can indicate the jurisdiction of issues concerning that address."
                    "Returns 'No information found' if unsuccessful."
                ),
            ),
            Tool(
                name="GetPolicyInfo",
                func=partial(self.get_policy_info),
                description=(
                    "Use this tool to obtain policy information from the municipality website that is related to the melding "
                    "in the format 'MELDING'. Returns 'No information found' if unsuccessful."
                ),
            ),
            Tool(
                name="GetCloseByReports",
                func=partial(self.get_duplicate_meldingen),
                description=(
                    "Use this tool to obtain a list of close-by reports (meldingen) which are possible duplicates"
                    "of a given report in the format 'MELDING'."# and an address in the format 'STRAATNAAM HUISNUMMER, POSTCODE"
                    "Such reports might indicate that the issue is already known and should not be reported again"
                ),
            ),
            Tool(
                name="GetSimilarReports",
                func=partial(self.get_similar_meldingen),
                description=(
                    "Use this tool to obtain a list of possibly similar reports (meldingen) together with their responses"
                    "to understand how similar cases from different locations were previously solved"
                ),
            ),
            Tool(
                name="GetLicensePlatePermitInfo",
                func=partial(self.get_license_plate_info),
                description=(
                    "If the report is about a wrongly parked car, always use this tool to find out whether a permit is"
                    "linked to the license plate (in the format 'LICENSE_PLATE') of that car."
                    "For example, that could a when a car is parked on the pavement "
                    " Returns 'No information found' if unsuccessful."
                ),
            ),
            Tool(
                name="GetPermitInfo",
                func=partial(self.get_noise_permit),
                description=("Use this tool find permits, for example for an event or that permit noise in a certain area for a certain period."
                            "A permit can indicate that for example an event is permitted or the noise from a complaint might be due to permitted noise."
                            "Returns 'No matching permit found.' if unsuccessful."
                ),
            )
        ]

        # Remove non-allowed tools
        tools = [tool for tool in tools if tool.name not in not_allowed_tools]
        return tools
    
    def initialize_agent_executor(self):
        """
        Initialize the agent executor with the specified tools and LLM.
        """
        prompt = self.create_custom_prompt()
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)

        # Initialize the agent with only the allowed tools
        agent = ZeroShotAgent(llm_chain=llm_chain, tools=self.tools)

        # Initialize the agent executor with the allowed tools
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors="Check your output and make sure it conforms."
        )
        return agent_executor

    def create_custom_prompt(self):
        """
        Create a custom prompt template for the agent using ZeroShotAgent.create_prompt.
        """
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

        # Prepare the input prompt variables for the agent
        melding_text = self.melding_attributes.get('MELDING', '')
        formatted_chat_history = get_formatted_chat_history(self.chat_history)
        melding_handling_guidelines = cfg.MELDING_HANDLING_GUIDELINES # This option allows for easy change of guidelines in config file
        # melding_handling_guidelines = open(os.path.join(cfg.MELDING_HANDLING_GUIDELINES_PATH, # this option for final repository
        #                                                     cfg.MELDING_HANDLING_GUIDELINES_FILE), 'r').read()

        inputs = {
            "melding": melding_text,
            "chat_history": formatted_chat_history,
            "date_time": self.get_date_time(),
            "melding_handling_guidelines": melding_handling_guidelines
        }

        # Run the agent with the input prompt
        try:
            response = self.agent_executor.invoke(inputs)['output']
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
            waste_collection_info = self.WasteCollectionTool.get_collection_times()
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
            gdf_bgt_info = self.BGTTool.get_bgt_features_at_coordinate()
            if gdf_bgt_info is not None and not gdf_bgt_info.empty:
                return self.BGTTool.get_functie_from_gdf(gdf_bgt_info)
            else:
                return "No information found"
        except Exception as e:
            logging.error(f"Failed to get BGT info: {e}")
            return "No information found"
        
    def get_address_owner_info(self, street: str) -> str:
        """
        Retrieve BGT (Basisregistratie Grootschalige Topografie) information based on the provided address.

        Args:
            address (str): The address in the format 'STRAATNAAM, HUISNUMMER, POSTCODE'.

        Returns:
            str: BGT information or 'No information found' if unsuccessful.
        """
        logging.info(f"Retrieving BGT info for address: {street}")
        try:
            return self.AddressOwnerTool.get_owner()
        except Exception as e:
            logging.error(f"Failed to get BGT info: {e}")
            return "No information found"
        
    def get_policy_info(self, melding: str) -> str:
        """
        Retrieve policy information from the website based on the provided melding.

        Args:
            melding (str): The melding done by the melder.

        Returns:
            str: Policy information or 'No information found' if unsuccessful.
        """
        logging.info(f"Retrieving policy info for melding: {melding}")
        try:
            policy_info = self.PolicyRetrieverTool.retrieve_policy()
            if not policy_info:
                return "No information found"
            return policy_info
        except Exception as e:
            logging.error(f"Failed to get policy info: {e}")
            return "No information found"

    def get_license_plate_info(self, license_plate: str) -> str:
        """
        Retrieve permit information based on the provided license plate.

        Args:
            license_plate (str): The license plate number.

        Returns:
            str: License plate permit information or 'No information found' if unsuccessful.
        """
        logging.info(f"Retrieving permit info for license plate: {license_plate}")
        try:
            license_plate_info = self.LicensePlatePermitTool.has_permit()
            if not license_plate_info:
                return "No information found"
            return license_plate_info
        except Exception as e:
            logging.error(f"Failed to get license plate permit info: {e}")
            return "No information found"

    def get_duplicate_meldingen(self, melding: str) -> list[str]:
        """
        Retrieve (possibly) duplicate meldingen

        Args:
            melding (str): The melding.
            address (str): The address in the format 'STRAATNAAM HUISNUMMER, POSTCODE'.

        Returns:
            [str]: a list of possible duplicates in the neighborhood.
        """
        logging.info(f"Retrieving duplicates of melding: {melding}")

        try:
            straatnaam = self.melding_attributes['STRAATNAAM']
            huisnummer = self.melding_attributes['HUISNUMMER']
            postcode = self.melding_attributes['POSTCODE']
            address = f"{straatnaam} {huisnummer}, {postcode}"
            melding = self.melding_attributes['MELDING']

            return self.MeldingenRetrieverTool.retrieve_meldingen(melding, address=address, top_k=5)

        except Exception as e:
            logging.error(f"Failed to retrieve relevant info: {e}")
            return "No information found"


    def get_similar_meldingen(self, melding: str) -> list[str]:
        """
        Retrieve related meldingen

        Args:
            melding (str): The melding.
            address (str): The address in the format 'STRAATNAAM HUISNUMMER, POSTCODE'.

        Returns:
            [str]: a list of relevant meldingen together with examples answers.
        """

        logging.info(f"Retrieving duplicates of melding: {melding}")

        try:
            melding = self.melding_attributes['MELDING']
            return self.MeldingenRetrieverTool.retrieve_meldingen(melding, top_k=5)

        except Exception as e:
            logging.error(f"Failed to retrieve relevant info: {e}")
            return "No information found"

    def get_noise_permit(self, melding: str) -> list[str]:
        """
        Retrieve noise permit.

        Args:
            melding (str): The melding.
            address (str): The address in the format 'STRAATNAAM HUISNUMMER, POSTCODE'.

        Returns:
            [str]: the permit text and some metadata distilled from the permit text
        """

        logging.info(f"Retrieving noise permit of: {melding}")

        logging.info(f"Retrieving policy info for melding: {melding}")

        try:
            permit_text = self.NoisePermitsTool.handle_melding(melding)
            if not permit_text:
                return "No information found"
            return permit_text
        except Exception as e:
            logging.error(f"Failed to get noise permit: {e}")
            return "No information found"

    def get_date_time(self) -> str:
        """
        Retrieve the current date and time in the specified format.
        
        Returns:
            str: Current date and time in the format "Wednesday 15 October 2024 18:45".
        """
        logging.info("Retrieving current date and time")
        return datetime.now().strftime("%A %d %B %Y %H:%M")


# Example usage:
if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    if cfg.track_emissions:
        tracker = EmissionsTracker(experiment_id = "inference_central_agentic_agent",
        co2_signal_api_token = sc.CO2_SIGNAL_KEY)
        tracker.start()


    melding_attributes = {

        # Example melding 1 (garbage collection)
        "MELDING": "Er ligt afval naast een container bij mij in de straat.",
        "STRAATNAAM": "Keizersgracht",
        "HUISNUMMER": "75",
        "POSTCODE": "1015CE",
        "LICENSE_PLATE_NEEDED": False,

        # Example melding 2 (parking permit)
        # "MELDING": "Er staat een auto geparkeerd op de stoep. Volgens mij heeft deze geen vergunning dus kunnen jullie deze wegslepen?",
        # "STRAATNAAM": "Keizersgracht",
        # "HUISNUMMER": "75",
        # "POSTCODE": "1015CE",
        # "LICENSE_PLATE_NEEDED": True,
        # "LICENSE_PLATE": "DC-743-SK"
    
        # Example melding 3 (noise permit)
        # "MELDING": "Er is erg veel lawaai van bouwwerkzaamheden bij station zuid, ook op zondag.",
        # "STRAATNAAM": "Zuidplein",
        # "HUISNUMMER": "136",
        # "POSTCODE": "1077XV",
        # "LICENSE_PLATE_NEEDED": False,

        # Example melding 4 (responsibility other party)
        # "MELDING": "Er ligt een gewonde duif op straat",
        # "STRAATNAAM": "Ertskade",
        # "HUISNUMMER": "164",
        # "POSTCODE": "1019BB",
        # "LICENSE_PLATE_NEEDED": False,

        # Example melding 5 (duplicate melding)
        # "MELDING": "Zwervers voor de Albert Heijn, klanten worden bang.",
        # "STRAATNAAM": "Wibautstraat",
        # "HUISNUMMER": "80",
        # "POSTCODE": "1091GP",
        # "LICENSE_PLATE_NEEDED": False,

        # Example melding 7 (policy)
        # "MELDING": "Mijn fiets is onterecht weggehaald, ik had hem 8 weken bij mij voor de deur staan. \
        # Nu moest ik geld betalen om hem op te halen. Ik wil dit geld terug.",
        # "STRAATNAAM": "Amsteldijk",
        # "HUISNUMMER": "10",
        # "POSTCODE": "1074HP",
        # "LICENSE_PLATE_NEEDED": False,

        # "MELDING": "Graffiti op Admiralengracht",
        # "STRAATNAAM": "Admiralengracht",
        # "HUISNUMMER": "107",
        # "POSTCODE": "1057VN",
        # "LICENSE_PLATE_NEEDED": False,

        # "MELDING": "grofvuil op de stoep",
        # "STRAATNAAM": "Tweede Kostverlorenkade",
        # "HUISNUMMER": "67",
        # "POSTCODE": "1053SB",
        # "LICENSE_PLATE_NEEDED": False,

        # "MELDING": "Er zijn weer schreeuwende jongeren op het Balboaplein. Ze zijn weer met vuurwerk aan het spelen.",
        # "STRAATNAAM": "Balboaplein",
        # "HUISNUMMER": "1",
        # "POSTCODE": "1057ET",
        # "LICENSE_PLATE_NEEDED": False,        
    }

    from helpers.llm_helpers import LLMRouter

    llm = LLMRouter.get_model(
        provider=cfg.provider,
        model_name=cfg.model_name,
        ollama_endpoint=cfg.OLLAMA_ENDPOINT,
        api_endpoint=cfg.AZURE_OPENAI_ENDPOINT,
        api_key=sc.API_KEY,
        api_version=cfg.AZURE_GPT_API_VERSION,
        hf_token=sc.HF_TOKEN,
        hf_cache=cfg.HUGGING_CACHE,
        params=cfg.params,
    )

    agent = CentralAgent(
        llm,
        chat_history=[],
        melding_attributes=melding_attributes,
    )
    agent.build_and_execute_plan()

    if cfg.track_emissions:
        tracker.stop()
