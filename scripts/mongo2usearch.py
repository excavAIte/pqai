"""
Create USearch indexes for documents pulled from a Mongo DB collection

This script can be modified for creating indexes of documents from any source
"""
import logging
import os
import sys

from pathlib import Path
import numpy as np
from tqdm import tqdm
from pymongo import MongoClient

from core import google_patent_scraper
from core.indexes import USearchIndex
from core.vectorizers import SentBERTVectorizer

MONGO_DBNAME = os.getenv("MONGO_DBNAME")
MONGO_PAT_COLL = os.getenv("MONGO_PAT_COLL")
MONGO_HOST = os.getenv("MONGO_HOST")
MONGO_PORT = int(os.getenv("MONGO_PORT", 27017))
MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")

BASE_DIR = str(Path(__file__).parent.parent.resolve())
INDEXES_DIR = "{}/indexes".format(BASE_DIR)

INDEX_NAME = "patent_data_"

sys.path.append(BASE_DIR)

vectorizer = SentBERTVectorizer()
N_DIMS = vectorizer.embed("sample string").shape[0]  # 384

def get_mongo_client():
    try:
        return MongoClient(
            host=MONGO_HOST,
            port=MONGO_PORT,
            username=MONGO_USER,
            password=MONGO_PASSWORD,
        )
    except Exception as e:
        logging.error("Failed to connect to MongoDB.", exc_info=e)
        sys.exit(1)


def fetch_and_vectorize_data_with_session(query, collection, vectorizer, client):
    try:
        N = collection.count_documents(query)
        labels = []
        vectors = np.zeros((N, vectorizer.embed("sample string").shape[0]))

        with client.start_session() as session:
            with collection.find(query, no_cursor_timeout=True, session=session) as cursor:
                for i, doc in enumerate(tqdm(cursor, total=N, desc="Processing documents")):
                    try:
                        dn = doc["bibliographic_information"]["publicationNumber"]
                        text = extract_text(doc)
                        vector = vectorizer.embed(text)
                        labels.append(dn)
                        vectors[i] = vector
                    except Exception as doc_error:
                        logging.error(
                            f"Error processing document {doc.get('bibliographic_information', {}).get('publicationNumber')}",
                            exc_info=doc_error,
                        )
        return vectors, labels
    except Exception as fetch_error:
        logging.error("Error fetching or processing data from MongoDB.", exc_info=fetch_error)
        sys.exit(1)


def extract_text(doc):
    text = [
        doc["bibliographic_information"]["title"],
        doc.get("abstract", ""),
    ]
    if "description" not in doc:
        scraper = google_patent_scraper.GooglePatentScraper(doc["bibliographic_information"]["publicationNumber"])
        text.append(scraper.get_patent_description())
    else:
        for entry in doc["description"]:
            text.extend([entry.get("title", ""), entry.get("description", "")])

    if "claims" in doc and doc["claims"]:
        claims_text = " ".join(claim.strip() for claim in doc["claims"])
        text.append(claims_text)

    return " ".join(t for t in text if t)

def save_vectors(vectors, year):
    try:
        np.save(INDEXES_DIR / f"{INDEX_NAME}_{year}_vectors.npy", vectors)
        logging.info("Vectors saved successfully.")
    except Exception as save_error:
        logging.error("Error saving vectors.", exc_info=save_error)
        sys.exit(1)

def create_index(vectors, labels, year):
    try:
        index = USearchIndex(name=INDEX_NAME + year)
        index._index_dir = INDEXES_DIR
        index.add_vectors(vectors, labels)
        logging.info(f"Index saved in {INDEXES_DIR}.")
    except Exception as index_error:
        logging.error("Error creating index.", exc_info=index_error)
        sys.exit(1)

def create_indices_from_mongo(start_date, end_date):
    client = get_mongo_client()
    collection = client[MONGO_DBNAME][MONGO_PAT_COLL]
    query = {"bibliographic_information.publicationDate": {"$gte": start_date, "$lte": end_date}}
    index_year = start_date[:4]
    logging.info("Fetching and processing data...")
    vectors, labels = fetch_and_vectorize_data_with_session(query, collection, vectorizer, client)

    logging.info("Saving vectors...")
    save_vectors(vectors, index_year)

    logging.info("Creating index...")
    create_index(vectors, labels, index_year)

if __name__ == "__main__":
    START_DATE = "20030101"
    END_DATE = "20031231"
    create_indices_from_mongo(START_DATE, END_DATE)
