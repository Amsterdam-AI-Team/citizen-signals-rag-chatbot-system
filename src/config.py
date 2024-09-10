# Constants
CHROMA_PATH = "../chroma"
DATA_PATH = '../data'
SESSION_FILE = "session.json"
ATTRIBUTES_FILE = "attributes.json"

ENDPOINT = 'azure' # set to 'local' if you wish to run locally using personal OpenAI API key

ENDPOINT_AZURE = "https://ai-openai-ont.openai.azure.com/"

API_KEYS = {
    "openai": "OpenAI API Key",
    "openai_azure": "OpenAI Azure API KEY"
}

model_dict = {"ChatGPT 4o": "gpt-4o"}

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
--------------------
MELDING: 
{melding}

--------------------

INSTRUCTIES:
Bepaal het type van de MELDING alleen als er voldoende specifieke en relevante informatie aanwezig is die overeenkomt met een bepaald probleem. Enkele voorbeelden van typen zijn: Kapotte straatverlichting, Zwerfvuil, Grofvuil, Kapotte bestrating, Beschadigde verkeersborden, Vervuilde grachten, Kapotte speeltoestellen, Illegale bouwwerkzaamheden, Overhangende takken, Foutgeparkeerde fietsen, Verkeershinder, Parkeerproblemen, Kapotte verkeerslichten, Overlast door vrachtverkeer, Verkeersonveilige situaties, Geluidsoverlast, Luchtvervuiling, Illegale bomenkap, Beschadiging van groenvoorzieningen, Huiselijk geweld of kindermishandeling, Dierenmishandeling, Overlast door personen, Drugsoverlast, Illegale wapenbezit, Brandveiligheid, Verwaarloosde panden, Muggen- en rattenoverlast, Asbest, Vervuilde openbare toiletten, Problemen met straatmeubilair, Meldingen over daklozen, Schending van coronamaatregelen, Overlast door evenementen, Illegale bijeenkomsten, Overvolle evenementen.
Je hoeft je niet te beperken tot deze voorbeelden; als je zelf iets beter vindt passen, mag dat ook.

Een type moet alleen worden toegewezen als:
1. De melding specifieke details bevat over het probleem.
2. De informatie duidelijk overeenkomt met een van de genoemde categorieën of een vergelijkbaar probleem.

Geef het bepaalde type terug als een JSON-object met de volgende structuur:
TYPE: type

Als de melding onvoldoende informatie bevat om een type te bepalen, geef dan een leeg JSON-object zonder key en value terug.
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