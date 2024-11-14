# Base path
BASE_PATH = '/home/azureuser/cloudfiles/code/blobfuse/meldingen'

# Constants
CHROMA_PATH = f'{BASE_PATH}/raw_data/amsterdam.nl/20241007_dump/chroma'
DOCUMENTS_PATH = f'{BASE_PATH}/raw_data/amsterdam.nl/20241007_dump/txt/scraped'
PERMITS_PATH = f'{BASE_PATH}/raw_data/permits/permits_related_to_license_plates/'
SESSION_FILE = "session.json"
ATTRIBUTES_FILE = "attributes.json"

HUGGING_CACHE = f"{BASE_PATH}/../hugging_cache"

FAISS_NOISE_PATH = f'{BASE_PATH}/raw_data/permits/permits_related_to_noise_disturbance/noise_permits_faiss_db'
METADATA_STORE_FILE = f'{BASE_PATH}/raw_data/permits/permits_related_to_noise_disturbance/noise_permits_faiss_metadata.json'

MELDING_HANDLING_GUIDELINES_PATH = f'{BASE_PATH}/raw_data/melding_handling_guidelines/'
MELDING_HANDLING_GUIDELINES_FILE = 'melding_handling_guidelines.txt'

# This is the actual folder with noise permit data
noise_permits_folder = f'{BASE_PATH}/raw_data/permits/permits_related_to_noise_disturbance/data'
# This is a subset of the noise permit data, for testing/dev
# noise_permits_folder = f'{BASE_PATH}/raw_data/permits//permits_related_to_noise_disturbance/data_sample'

# Main folders
meldingen_in_folder = f'{BASE_PATH}/raw_data'
meldingen_out_folder = f'{BASE_PATH}/processed_data/'

source = "20240821_meldingen_results_prod"
meldingen_dump = f"{meldingen_in_folder}/{source}.csv"

index_storage_folder = f"{meldingen_out_folder}/indices"

track_emissions = True # Set to True to track emissions to 

embedding_model_name = "intfloat/multilingual-e5-large"
# embeddng_model_name = "jegormeister/bert-base-dutch-cased-snli"
# embeddng_model_name = "NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers"

ENDPOINT = 'azure' # set to 'local' if you wish to run locally using personal OpenAI API key

ENDPOINT_AZURE = "https://ai-openai-ont.openai.azure.com/"

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

Bepaal of de melding duidelijk aangeeft wat het probleem is, en of deze concreet genoeg is om door de verantwoordelijke werknemers opgepakt te worden. 
Losse woorden of alleen steekwoorden zijn bijvoorbeeld onvoldoende concreet.
Als het voldoende concreet is, wijs een specifiek onderwerp toe dat het probleem beschrijft. Kies een onderwerp dat relevant is voor gemeentelijke diensten, zoals:
- Vuilnis en afval
- Openbare ruimte (zoals parken, trottoirs, straatmeubilair)
- Verkeer en parkeren
- Straatverlichting
- Overlast (geluid, bouw, enz.)
- Water en riolering

Geef het type van de melding als een JSON-object in de volgende structuur:
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
{melding}

The chat history is:
{chat_history}

The current date and time is:
{date_time}

Your goal is to create a plan to retrieve and format information that could be shared with the melder (citizen), such that the melding does not need to be escalated into the melding system.

General Policies to Follow:
{melding_handling_guidelines}

If you find useful information using the tools, provide only that information in your Final Answer (in the language of the melding) while adhering to the above policies.
If you cannot find any useful information, respond with: "Geen bruikbare informatie gevonden. Je melding wordt in ons systeem gezet om op te lossen."
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

MELDING_HANDLING_GUIDELINES = """
Garbage Collection
- If the report concerns garbage beside a container or an overflowing container or anything similar, and the GetWasteCollectionInfo and current date and time confirm that \
    collection is scheduled for the same day, inform the reporter that the garbage is likely to be collected later that day. If collection isn't scheduled \
      until another day, inform the user that the garbage will not be picked up soon according to the regular pickup schedule, and the report will therefore \
        be forwarded to the system for further handling.

Check for Duplicate Meldingen
- If the GetDuplicateMeldingen tool indicates that duplicates are found, let the reporter know that the issue has already been \
    noted and is being addressed. A melding is only considered duplicate if it exactly matches an existing one.

Non-Municipal Areas
- If the GetBGTInfo tool indicates that the "bgt functie" of the address is not classified as a "pand" (building), inform the reporter that the issue \
    is outside the municipality's jurisdiction and should be reported to the appropriate party.

Policy-Specific Responses
- If the GetPolicyInfo tool highlights relevant policy regarding the issue, inform the reporter accordingly. For instance, if the report concerns \
    a shrub partially obstructing the pavement, let them know that municipal intervention will only occur if the obstruction poses a safety hazard, as per policy. \

Noise Disturbances
- If the HandleNoiseComplaint tool returns a permit, please return structured information to the reporter about the permit in the format as is returned by the tool. \
    Feel free to be a bit verbose, and please explain why the noise may or may not be permitted, \
        and if anything about objection to the permit is mentioned, please return this as well. \

License Plate Permits
- If the GetLicensePlatePermitInfo returns that a car has a permit, it is allowed to park on the pavement/sidewalk \
    If, in that case, the report is about a car that is parked on the pavement/sidewalk, please notify the reporter that this specific car is permitted to do this.\
        Do notify them that this is only the case for shorter periods (of up to a couple of hours) and not for (for example) multiple days.

General Rules
- Always use the GetDuplicateMeldingen tool to check wether duplicate meldingen exist.
- Always use the GetBGTInfo to check if the melding is being made on municipality's jurisdiction.
- Don't use salutation and closing in your final answer.
"""
