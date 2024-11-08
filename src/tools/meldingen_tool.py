"""
Implementation of a meldingen retrieval tool.
The tool is used by the central agent to retrieve duplicate or similar meldingen.
"""
import logging
import os
import pickle
from ast import literal_eval
from pathlib import Path
from pprint import pprint

import geopandas as gpd
import pandas as pd
import requests
from pyproj import Transformer
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search
from shapely.geometry import Point
from codecarbon import EmissionsTracker
import config as cfg


class MeldingenRetrieverTool:
    """
    A class to retrieve related meldingen.
    If provided with an address, it attempts to retrieve duplicate meldingen from neighborhood.
    Else, it retrieves simply similar meldingen from any part of town.

    Attributes:
        model_name (str): model ID of the desired embedding model
        meldingen_dump (str): path to csv file with meldingen to be indexed
        index_storage_folder (str): path to folder where pre-computed indices are stored
    """

    def __init__(self, model_name, meldingen_dump, index_storage_folder):
        """Initializes a MeldingenRetrieverTool"""
        self.model_name = model_name
        self.meldingen_path = Path(meldingen_dump)
        self.index_storage_folder = Path(index_storage_folder)

        self.embedding_model = self._load_model()
        self._load_meldingen()
        self._init_index()

    def _load_meldingen(self):
        """Load existing meldnigen. Set coordnates and prep geodf."""
        self.meldingen_data = pd.read_csv(self.meldingen_path)
        self.meldingen_data["location.geometrie.coordinates"] = self.meldingen_data[
            "location.geometrie.coordinates"
        ].apply(literal_return)
        self.meldingen_data = self.meldingen_data[
            ~self.meldingen_data["location.geometrie.coordinates"].isna()
        ]
        self.meldingen_data["created_at"] = pd.to_datetime(self.meldingen_data["created_at"])
        self.meldingen_data["updated_at"] = pd.to_datetime(self.meldingen_data["updated_at"])

        self.meldingen_data[["lat", "lon"]] = pd.DataFrame(
            self.meldingen_data["location.geometrie.coordinates"].to_list(),
            index=self.meldingen_data.index,
        )
        self.meldingen_geodata = gpd.GeoDataFrame(
            self.meldingen_data,
            geometry=gpd.points_from_xy(self.meldingen_data["lat"], self.meldingen_data["lon"]),
            crs="epsg:4326",
        )

    def _get_persist_path(self):
        """Get path to index. Embed number of meldingen to account for different sizes."""
        # TODO: Incorporate hashing to reindex on document update
        return (
            self.index_storage_folder
            / f"meldingen_{len(self.meldingen_data)}_{self.model_name.replace('/', '_')}.pkl"
        )

    def _load_model(self):
        logging.info(f"Loading {self.model_name}...")
        embedding_model = SentenceTransformer(self.model_name, trust_remote_code=True)
        return embedding_model

    def _init_index(self):
        self.index_persist_path = self._get_persist_path()
        logging.info(f"Known persist path: {self.index_persist_path}")

        if self.index_persist_path.exists():
            logging.info("Loading existing corpus embeddings...")
            with open(self.index_persist_path, "rb") as persist_file:
                dump = pickle.load(persist_file)
                self.documents = dump["documents"]
                self.corpus_embeddings = dump["embeddings"]
        else:
            logging.info(f"Embedding documents using {self.model_name}...")
            # self._set_documents()
            documents = self.meldingen_data["text"].values
            self.corpus_embeddings = self.embedding_model.encode(documents, show_progress_bar=True)

            self.index_persist_path.parents[0].mkdir(parents=True, exist_ok=True)
            with open(self.index_persist_path, "wb") as persist_file:
                pickle.dump(
                    {"documents": documents, "embeddings": self.corpus_embeddings}, persist_file
                )

    def filter_meldingen_embeddings(self, address):
        """Filter meldingen around a certain address"""
        transformer_to_rd = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
        lat, lon = get_lat_lon_from_address(address)
        poi = Point(transformer_to_rd.transform(lat, lon))
        self.meldingen_geodata["distance"] = self.meldingen_geodata.to_crs("EPSG:28992").distance(
            poi
        )

        # Simply take all within 100m
        # indices = self.meldingen_geodata["distance"] < 100
        # meldingen = self.meldingen_data[indices]

        # Take 5 closest per category and filter those of them which are within 100
        closest_per_category = self.meldingen_geodata.groupby(
            "category.sub", group_keys=False
        ).apply(lambda x: x.nsmallest(5, ["distance"]))
        indices = closest_per_category[closest_per_category["distance"] < 100].index.to_list()

        meldingen = self.meldingen_data.iloc[indices]
        embeddings = self.corpus_embeddings[indices]

        return meldingen, embeddings

    def retrieve_meldingen(self, query, top_k=5, address=None):
        """Retrieve a number of (top_k) duplicate or similar meldingen."""
        logging.info(f"Retrieving top {top_k} for query: {query}")
        logging.info(f"Known address: {address}")

        if address:
            meldingen_data, corpus_embeddings = self.filter_meldingen_embeddings(address)

        else:
            meldingen_data = self.meldingen_data
            corpus_embeddings = self.corpus_embeddings

        query_embedding = self.embedding_model.encode(query)
        hits = semantic_search(query_embedding, corpus_embeddings, top_k=top_k)

        # TODO: cutoff based on similarity?
        hit_ids = [hit["corpus_id"] for hit in hits[0]]
        logging.info(f"Hits: {hits}")

        # docs = meldingen_data.iloc[hit_ids]["text"].values.tolist()
        docs = (
            meldingen_data.iloc[hit_ids]
            .apply(
                lambda x: (
                    f"ADDRESS: {x['location.address_text']}\n"
                    f"CATEGORY: {x['category.main']}, {x['category.sub']}\n"
                    f"MELDING: {x['text']}\n"
                    f"RESPONSE: {x['status.text']}\n"
                ),
                axis=1,
            )
            .to_list()
        )

        return docs


def literal_return(val):
    try:
        return literal_eval(val)
    except ValueError:
        return val


# TODO: Transfer to common utils
def get_lat_lon_from_address(address):
    """
    Retrieves the longitude and latitude for a given address using the Nominatim API.

    Args:
        address (str): The address to geocode.

    Returns:
        tuple: A tuple containing the longitude and latitude, or (None, None) if not found.
    """
    # Define the endpoint and parameters for the Nominatim API
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": address, "format": "json", "limit": 1}

    # Include the User-Agent header
    headers = {"User-Agent": "BGTFetcher/1.0 (test@test.com)"}

    # Make a GET request to the API with headers
    response = requests.get(url, params=params, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        if data:
            lat = float(data[0]["lat"])
            lon = float(data[0]["lon"])
            return lon, lat
        else:
            print("No results found for the address.")
            return None, None
    else:
        print(f"Error fetching coordinates for address: {response.status_code}")
        return None, None


if __name__ == "__main__":
    logging.basicConfig(level="INFO")

    # TODO: import all from cfg
    HUGGING_CACHE = "/home/azureuser/cloudfiles/code//hugging_cache"
    os.environ["TRANSFORMERS_CACHE"] = HUGGING_CACHE
    os.environ["HF_HOME"] = HUGGING_CACHE

    meldingen_in_folder = "/home/azureuser/cloudfiles/code/blobfuse/meldingen/raw_data"
    meldingen_out_folder = "/home/azureuser/cloudfiles/code/blobfuse/meldingen/processed_data/"
    source = "20240821_meldingen_results_prod"
    meldingen_path = f"{meldingen_in_folder}/{source}.csv"
    index_folder = f"{meldingen_out_folder}/indices"

    model_name = "intfloat/multilingual-e5-large"
    # model_name = "zeta-alpha-ai/Zeta-Alpha-E5-Mistral"
    # model_name = "jegormeister/bert-base-dutch-cased-snli"
    # model_name = "GroNLP/bert-base-dutch-cased"
    # model_name = "Alibaba-NLP/gte-multilingual-base"
    # model_name = "NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers"
    # model_name = "sentence-transformers/all-mpnet-base-v2"

    # Specify the address
    address = "Admiralengracht 100, 1057ET"  # Replace with your desired address

    # Instantiate the MeldingenRetrieverTool class with the address
    retriever = MeldingenRetrieverTool(model_name, meldingen_path, index_folder)

    # Select/Write a melding
    melding = "vuilnis ligt er al dagen"
    # melding = "Overlast door zwervers naast winkel. Heel onveilig situatie. Medewerkers en ons personeel zijn bang."
    # melding = "Graffiti op de brug, andere kant van waar net andere graffti verwjiderd was."

    print("===== With an address =====")
    duplicate = retriever.retrieve_meldingen(melding, address=address, top_k=10)
    pprint(list(enumerate(duplicate)))

    print("===== Without an address =====")
    similar = retriever.retrieve_meldingen(melding, top_k=10)
    pprint(list(enumerate(similar)))
