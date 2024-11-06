import os
import pdfplumber
import re
import ast
import requests
from langchain.prompts import ChatPromptTemplate
import config as cfg
from pathlib import Path
import faiss
import numpy as np
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
import json
from helpers.embedding_helpers import OpenAIEmbeddingFunction
from codecarbon import EmissionsTracker, track_emissions

# Initialize FAISS index filepath
FAISS_INDEX_PATH = cfg.FAISS_NOISE_PATH 
METADATA_STORE_FILE = cfg.METADATA_STORE_FILE

# FAISS_INDEX_PATH = os.path.join(os.getcwd(), "faiss/faiss_index_test")
# METADATA_STORE_FILE =os.path.join(os.getcwd(), "faiss/noise_permits_faiss_metadata.json")

# Load existing metadata or initialize an empty store
try:
    with open(METADATA_STORE_FILE, 'r') as f:
        metadata_store = json.load(f)
    print("Loaded metadata from disk.")
except:
    metadata_store = {}

# Step 1: Extract text from each PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text
print(cfg.API_KEYS['openai_azure'])

# Step 2: Use an LLM to extract metadata from the permit text
metadata_prompt = ChatPromptTemplate.from_template("""
Extract the location, date issued, permit_validity_time_window, and type of permit, case number, and description from the following text:
Please give the location in the format:["Streetname",  "housenumber", "zipcode (in 9999AA format)", "cityname"]. If any is unknown, please use "Onbekend".
Text: {permit_text}
Location:
Date issued:
Permit validity time window:
Type of Permit:
Case number:
Description:
""")

def initialize_llm():
        """
        Initialize the language model based on the configuration.
        """
        if cfg.ENDPOINT == 'local':
            llm = ChatOpenAI(model_name='gpt-4o',
                api_key=cfg.API_KEYS["openai"], 
                temperature=0
            )
        elif cfg.ENDPOINT == 'azure':
            llm = AzureChatOpenAI(
                deployment_name='gpt-4o',
                model_name='gpt-4o',
                azure_endpoint=cfg.ENDPOINT_AZURE,
                api_key=cfg.API_KEYS["openai_azure"],
                api_version="2024-02-15-preview",
                temperature=0,
            )
        print(f"The OpenAI LLM is using model: {llm.model_name}")
        return llm

llm_chain = metadata_prompt | initialize_llm()

def extract_data(permit_text):
    data = llm_chain.invoke(permit_text).content
    return data

# Step 3: Extract metadata fields from the LLM response
def extract_metadata(data):
    metadata_lines = data.split("\n")
    location = metadata_lines[0].split(":")[1].strip()
    date = metadata_lines[1].split(":")[1].strip()
    permit_time_window = metadata_lines[2].split(":")[1].strip()
    permit_type = metadata_lines[3].split(":")[1].strip()
    case_number = metadata_lines[4].split(":")[1].strip()

    return {
        "location": location,
        "date_issued": date,
        "permit_time_window": permit_time_window,
        "permit_type": permit_type,
        "case_number": case_number
    }

# Step 4: Process address metadata using an external API
def process_location_metadata(location):
    pattern = r'\[.*?\]'
    match = re.search(pattern, location)
    if match:
        result = match.group()
        result_list = ast.literal_eval(result)
        result_dict = dict(zip(['Straatnaam', 'Huisnummer', 'Postcode', 'Stad'], result_list))
    else:
        result_dict = {k:'Onbekend' for k in ['Straatnaam', 'Huisnummer', 'Postcode', 'Stad']}

    if result_dict and result_dict['Postcode'] != 'Onbekend':
        params = {'postcode': result_dict['Postcode']}
    elif result_dict and result_dict['Straatnaam'] != 'Onbekend':
        params = {'openbareruimteNaam': result_dict['Straatnaam']}
    else:
        return "No postcode or straatname found"
    
    url = "https://api.data.amsterdam.nl/v1/dataverkenner/bagadresinformatie/bagadresinformatie/"
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if data:
            try:
                return data['_embedded']['bagadresinformatie'][0]['gebiedenStadsdeelNaam']
            except:
                return "No results found for the address."
    else:
        return f"Error fetching coordinates for address: {response.status_code}"

# Step 5: Load or initialize the FAISS index
def load_or_initialize_faiss_index(dimension):
    if os.path.exists(FAISS_INDEX_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        print("Loaded FAISS index from disk.")
    else:
        index = faiss.IndexFlatL2(dimension)
        print("Initialized a new FAISS index.")
    return index

# Step 6: Populate FAISS with the processed PDF data and metadata
def store_in_faiss(permit_text, metadata):
    embedder = OpenAIEmbeddingFunction()
    embedding = embedder.embed_query(permit_text)
    embedding_array = np.array(embedding).reshape(1, -1).astype('float32')

    # Load or initialize the FAISS index
    dimension = embedding_array.shape[1]
    global index
    if 'index' not in globals() or index.d != dimension:
        index = load_or_initialize_faiss_index(dimension)

    # Add the embedding to the FAISS index
    faiss_id = index.ntotal  # This is the FAISS ID before the next add operation
    index.add(embedding_array)
    
    # Save metadata with FAISS ID
    metadata_store[faiss_id] = metadata

    # Save the updated index and metadata to disk
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_STORE_FILE, 'w') as f:
        json.dump(metadata_store, f)
    print(f"FAISS index and metadata saved to disk.")


# Step 7: Main function to process a folder of PDFs
def process_pdf_folder(pdf_folder_path):
    for pdf_file in os.listdir(pdf_folder_path):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder_path, pdf_file)

            # Step 1: Extract text from the PDF
            permit_text = extract_text_from_pdf(pdf_path)

            # Step 2: Extract structured data from the PDF using LLM
            extracted_data = extract_data(permit_text)

            # Step 3: Extract metadata from the LLM response
            try:
                metadata = extract_metadata(extracted_data)
            except:
                metadata = {
                        "location": "Unkown",
                        "date_issued":"Unkown",
                        "permit_time_window":"Unkown",
                        "permit_type": "Unkown",
                        "case_number": "Unkown",
                        }


            metadata['description'] = permit_text

            # Step 4: Process metadata using external APIs
            location_metadata = process_location_metadata(metadata['location'])
            metadata['gebied'] = location_metadata  # Add additional metadata

            # Step 5: Store the text and metadata in FAISS
            store_in_faiss(extracted_data, metadata)

    print("All PDF data has been processed and stored in FAISS.")

# Example usage: Process a folder of PDFs
pdf_folder_path = cfg.noise_permits_folder

# Dependent on what we specify in .config we either want to run codecarbon or not
if cfg.track_emissions:
    tracker = EmissionsTracker(experiment_id = "oneoff_populate_permit_db_faiss")
    tracker.start()
    process_pdf_folder(pdf_folder_path)
    tracker.stop()
else:
    process_pdf_folder(pdf_folder_path)