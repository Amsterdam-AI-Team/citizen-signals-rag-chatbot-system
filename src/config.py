# Base path
BASE_PATH = '/home/azureuser/cloudfiles/code/blobfuse/meldingen'

# Constants
CHROMA_PATH = f'{BASE_PATH}/raw_data/amsterdam.nl/20241007_dump/chroma'
DOCUMENTS_PATH = f'{BASE_PATH}/raw_data/amsterdam.nl/20241007_dump/txt/scraped'
PERMITS_PATH = f'{BASE_PATH}/raw_data/permits/permits_related_to_license_plates/'
ADDRESS_OWNERS_PATH = f'{BASE_PATH}/raw_data/address_owners/'
SESSION_FILE = "session.json"
ATTRIBUTES_FILE = "attributes.json"

HUGGING_CACHE = "/home/azureuser/cloudfiles/code/hugging_cache/"

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


gpt_params = {
    "frequency_penalty": 0,
    "presence_penalty": 0,
    # "stop": None,
    "temperature": 0.25,
}

hf_params = {
    "do_sample": True,
    "temperature": 0.2,
    "top_p": 0.8,
    "top_k": 25,
    "max_new_tokens": 200,
    "no_repeat_ngram_size": 3,
    "num_return_sequences": 1,
}


# Config to run using GPT
provider = "azure"
model_name = "gpt-4o"
params = gpt_params

# Config to run using ollama (hardly tested, requires running model first)
# provider = "ollama"
# model_name = "mistral:instruct"
# params = hf_params

# Config to run using an open model using HuggingFace & transformers
# provider = "huggingface"
# model_name = "mistral-7b-instruct"
# # model_name = "mixtral-7b-instruct"
# # model_name = "llama-13b-chat"
# # model_name = "llama3-8b-instruct"
# params = hf_params


OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"

AZURE_OPENAI_ENDPOINT = "https://ai-openai-ont.openai.azure.com/"
AZURE_GPT_API_VERSION = "2024-02-15-preview"
AZURE_ADA_API_VERSION = "2024-06-01"

ENDPOINT_BAG = "https://api.data.amsterdam.nl/v1/dataverkenner/bagadresinformatie/bagadresinformatie/"

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
You are an AI assistant tasked with helping citizens with the following incident report (melding):
{melding}

The chat history is:
{chat_history}

The current date and time is:
{date_time}

Your need to create a plan to retrieve and format information that could be shared with the melder (citizen). 
Your goal is providing citizens with relevant information from the municipality regarding the incident report of the citizen. 
If possible, you want to provide the citizen with helpful information such that they do not have to speak to an employee of the municipality. 

General Policies to Follow:
{melding_handling_guidelines}

If you find useful information using the tools, provide that information in your Final Answer (in the language of the melding) while adhering to the above policies. 
Start with "Ik heb je opmerking verzonden. Dit is wat ik je nu al kan vertellen." (in the language of the melding).
If you cannot find any useful information, respond in your Final Answer with: "Ik heb je opmerking verzonden. Helaas heb ik geen relevante informatie gevonden om alvast met je te delen" (in the language of the melding).
Don't thank the citizen for making a melding. Do thank them for being involved.
"""

AGENTIC_AI_AGENT_PROMPT_FORMAT_INSTRUCTIONS = """
Use the following format:

Question: the melding to be resolved
Thought: your reasoning about what to do
Action: the action to take
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I have gathered all possible information.
Draft: a first version of the response to the melder
Action: re-write the draft to create a final answer that doesn't share privacy sensitive information (like names or addresses).
Final Answer: the response to the melder

When responding, output the response in one of two formats:

**Option 1:**
If you want to use a tool, respond with:
Action: the action to take; This must be always one of [{tool_names}]. Do not come up with any other tools
Action Input: the input to the action
Example:
Action: GetDuplicateMeldingen
Action Input: De gras in Erasmuspark is te hoog, het is gevaarlijk voor mijn kindje. Kan de gemeente dat gewoon vaker snoeien?!

**Option #2:**
If you want to respond directly, respond with:
> Final Answer: ...

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

Check for Possibly Duplicate Reports
- Always use the GetCloseByReports tool to identify possibly duplicate reports. Assess yourself if the information is correct and only consider \
    duplicates if you are certain without doubt that the location and issue are exactly the same. \
    If full duplicates are found, try to confirm the resolution by e.g. checking the garbage pick up times or municipal policy. \
    Then let the reporter know that the issue has already been noted and is being addressed. \

Non-Municipal Areas
- If the GetAddressOwnerInfo tool indicates that the owner of the address is Prorail, NS or any other non-municipal organisation, inform the reporter that the issue \
    is outside the municipality's jurisdiction and should be reported to the appropriate party.

Policy-Specific Responses
- If the GetPolicyInfo tool highlights relevant policy regarding the issue, inform the reporter accordingly. For instance, if the report concerns \
    a shrub partially obstructing the pavement, let them know that municipal intervention will only occur if the obstruction poses a safety hazard, as per policy. \

Noise Disturbances
- If the HandleNoiseComplaint tool returns a permit, then assess yourself whether this information is relevant and only consider the permit \
    if it is related to the exact same location and type of noise reported. \
    Then please return structured information to the reporter about the permit in the format as is returned by the tool. \
    Feel free to be a bit verbose, and please explain why the noise may or may not be permitted, \
        and if anything about objection to the permit is mentioned, please return this as well. \

License Plate Permits
- If the GetLicensePlatePermitInfo returns that a car has a permit, it is allowed to park on the pavement/sidewalk \
    If, in that case, the report is about a car that is parked on the pavement/sidewalk, please notify the reporter that this specific car is permitted to do this.\
        Do notify them that this is only the case for shorter periods (of up to a couple of hours) and not for (for example) multiple days.

General Rules
- Always use the GetDuplicateMeldingen tool to check whether duplicate meldingen exist.
- Always use the GetBGTInfo to check if the melding is being made on municipality's jurisdiction.

Answering Rules
- Never tell a citizen that they have to fix the problem themselves.
- Don't mention planned actions for which there is no proof (a  mention in a previous melding is not enough proof). 
"""
