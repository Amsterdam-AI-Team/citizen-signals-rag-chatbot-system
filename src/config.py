# Constants
CHROMA_PATH = '/home/azureuser/cloudfiles/code/blobfuse/meldingen/raw_data/amsterdam.nl/20241007_dump/chroma'
DOCUMENTS_PATH = '/home/azureuser/cloudfiles/code/blobfuse/meldingen/raw_data/amsterdam.nl/20241007_dump/txt/scraped'
PERMITS_PATH = '/home/azureuser/cloudfiles/code/blobfuse/meldingen/raw_data/permits/permits_related_to_license_plates/'
SESSION_FILE = "session.json"
ATTRIBUTES_FILE = "attributes.json"

HUGGING_CACHE = "/home/azureuser/cloudfiles/code//hugging_cache"

FAISS_NOISE_PATH = '/home/azureuser/cloudfiles/code/blobfuse/meldingen/raw_data/permits/permits_related_to_noise_disturbance/noise_permits_faiss_db'
METADATA_STORE_FILE = '/home/azureuser/cloudfiles/code/blobfuse/meldingen/raw_data/permits/permits_related_to_noise_disturbance/noise_permits_faiss_metadata.json'

# This is the actual folder with noise permit data
# noise_permits_folder = "/home/azureuser/cloudfiles/code/blobfuse/meldingen/raw_data/permits/permits_related_to_noise_disturbance/data"
# This is a subset of the noise permit data, for testing/dev
noise_permits_folder = "/home/azureuser/cloudfiles/code/blobfuse/meldingen/raw_data/permits//permits_related_to_noise_disturbance/data_sample"

# Main folders
meldingen_in_folder = "/home/azureuser/cloudfiles/code/blobfuse/meldingen/raw_data"
meldingen_out_folder = "/home/azureuser/cloudfiles/code/blobfuse/meldingen/processed_data/"
amsterdam_nl_scrapped_folder = "/home/azureuser/cloudfiles/code/blobfuse/meldingen/amsterdam_nl_scraper/amsterdam_nl_data/html/scraped/"

source = "20240821_meldingen_results_prod"
meldingen_dump = f"{meldingen_in_folder}/{source}.csv"

index_storage_folder = f"{meldingen_out_folder}/indices"

track_emissions = True # Set to True to track emissions to 

embedding_model_name = "intfloat/multilingual-e5-large"
# embeddng_model_name = "jegormeister/bert-base-dutch-cased-snli"
# embeddng_model_name = "NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers"

ENDPOINT = 'azure' # set to 'local' if you wish to run locally using personal OpenAI API key

ENDPOINT_AZURE = "https://ai-openai-ont.openai.azure.com/"

API_KEYS = {
    "openai": "OpenAI API KEY",
    "openai_azure": "d9417c3021824d9a9825c8aa80297624"
    # "openai_azure": "OpenAI Azure API KEY"
}

model_dict = {"ChatGPT 4o": "gpt-4o"}
summarize_melding_for_policy_retrieval = False # set to True if you wish to summarize melding for policy retrieval

SYSTEM_CONTENT_INITIAL_RESPONSE = "Je bent een behulpzame en empathische probleemoplosser. \
            Je doel is om bewoners van Amsterdam te ondersteunen door begripvolle en respectvolle reacties te geven op hun meldingen en klachten. \
                Toon altijd begrip voor de zorgen en gevoelens van de melder, en reageer op een manier die hen het gevoel geeft gehoord en serieus genomen te worden."

SYSTEM_CONTENT_ATTRIBUTE_EXTRACTION = "Je bent een behulpzame probleemoplosser. \
            Je doel is om bewoners van Amsterdam te ondersteunen door specifieke details te extraheren uit meldingen. \
                Reageer met de gevraagde informatie in een duidelijk gestructureerd JSON-formaat."

INITIAL_MELDING_TEMPLATE = """
--------------------
MELDING: 
{melding}

--------------------
TYPE MELDING: 
{type}

--------------------

INSTRUCTIES:
Schrijf een passende en empathische eerste reactie op deze MELDING. 
Toon begrip voor de situatie en de gevoelens van de melder. Gebruik eventueel de informatie uit TYPE MELDING om de reactie relevanter te maken. 
Houd de reactie kort en bondig, zonder aan- of afkondiging, en vermijd het noemen van eventuele vervolgstappen.
"""

MELDING_TYPE_TEMPLATE = """
Melding: {melding}

1. Analyseer de melding: 
Bepaal of de melding duidelijk aangeeft wat het probleem is, en of deze concreet genoeg is om door de verantwoordelijke werknemers opgepakt te worden.
- Is het probleem specifiek en gedetailleerd omschreven?

2. Onderwerp toewijzen:
Als de melding voldoet aan bovenstaande criteria, wijs een specifiek onderwerp toe dat het probleem beschrijft. Kies een onderwerp dat relevant is voor gemeentelijke diensten, zoals:
- Vuilnis en afval
- Openbare ruimte (zoals parken, trottoirs, straatmeubilair)
- Verkeer en parkeren
- Straatverlichting
- Overlast (geluid, bouw, enz.)
- Water en riolering

3. Outputformaat:
Als een onderwerp kan worden toegewezen, geef dan het type van de melding als een JSON-object in de volgende structuur:
TYPE: type

Als er onvoldoende informatie is, geef dan een leeg JSON-object zonder key en value terug.
"""

MELDING_ADDRESS_TEMPLATE = """
--------------------
GESPREKSGESCHIEDENIS: 
{history}

--------------------
MELDING: 
{melding}

--------------------

INSTRUCTIES:
Bepaal of er een adres gegevens zijn vermeld in de GESPREKSGESCHIEDENIS en/of MELDING. 
Adres gegevens zijn: STRAATNAAM, HUISNUMMER, en POSTCODE.
Een POSTCODE is alleen correct als het een van de volgende formatteringen heeft: AAAA11, AAAA 11.

Geef gevonden adresgegevens terug als een JSON-object met de volgende velden:
STRAATNAAM: straatnaam of leeg als niet aanwezig
HUISNUMMER: huisnummer of leeg als niet aanwezig
POSTCODE: postcode of leeg als niet aanwezig
"""

LICENSE_PLATE_TEMPLATE = """
--------------------
GESPREKSGESCHIEDENIS: 
{history}

--------------------
MELDING: 
{melding}

--------------------

INSTRUCTIES:
Bepaal of er een kenteken nummer vermeld is in de GESPREKSGESCHIEDENIS en/of MELDING.

Geef een gevonden kenteken nummer terug als een JSON-object met de volgende velden:
LICENSE_PLATE: kenteken

Als er onvoldoende informatie is, geef dan een leeg JSON-object zonder key en value terug.
"""

AGENTIC_AI_AGENT_PROMPT_PREFIX = """
You are an AI assistant tasked with resolving the following melding (incident report) from a citizen:
"{melding}"

The chat history is:
"{chat_history}"

Your goal is to create a plan to retrieve and format information that could be shared with the melder (citizen), such that the melding does not need to be escalated into the melding system.

If you find useful information using the tools, provide that information in your Final Answer.

If you cannot find any useful information, respond with:
"No useful information was found. Your melding will be escalated in the system."
"""

AGENTIC_AI_AGENT_PROMPT_FORMAT_INSTRUCTIONS = """
Use the following format:

Question: the melding to be resolved
Thought: your reasoning about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I have gathered all possible information.
Final Answer: the response to the melder
"""

AGENTIC_AI_AGENT_PROMPT_SUFFIX = """
Begin!

Question: {melding}
Thought: {agent_scratchpad}
"""

RESTAFVAL_COLLECTION_INFO_RAG_TEMPLATE = """
--------------------
AFVALWIJZER INFORMATIE:
{waste_info}

--------------------
MELDING:
{melding}

--------------------
TYPE AFVAL:
{subtype}

--------------------
HUIDIGE DATUM EN TIJD:
{date_time}

--------------------

INSTRUCTIES:
Controleer of de AFVALWIJZER INFORMATIE relevante details bevat met betrekking tot het TYPE AFVAL die kunnen helpen bij het oplossen van de MELDING.
Het is belangrijk dat de melder van het probleem niet als veroorzaker van het probleem wordt gezien.
Zorg dat het bericht een geschikte toon heeft, rekening houdend met de MELDING. 
Geef ALLEEN de informatie terug die voorkomt uit het stappenplan omdat het onderdeel is van een lopend gesprek.
"""

SUMMARIZE_MELDING_TEMPLATE = """

MELDING: {melding}
--------------------

INSTRUCTIES:
Geef een samenvatting van de MELDING als een zoekterm.
Geef ALLEEN deze geparafraseerde zoekterm terug.
"""

POLICY_MELDING_TEMPLATE = """
--------------------
DOCUMENTEN: 
{context}
--------------------


MELDING: {melding}
--------------------

INSTRUCTIES:
Kijk of er informatie in de DOCUMENTEN staat die van pas kan komen bij het oplossen van de melding.
Dat zou bijvoorbeeld algemene informatie of beleid over het onderwerp kunnen zijn.
Houd je antwoorden gegrond in de inhoud in de DOCUMENTEN.
Je mag ook eventuele links meegegeven die leiden naar de webpagina waar het antwoord te vinden is.
"""
