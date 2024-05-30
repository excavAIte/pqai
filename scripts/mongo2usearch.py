"""
Create USearch indexes for documents pulled from a Mongo DB collection

This script can be modified for creating indexes of documents from any source
"""

# import os
import sys

# import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
from pymongo import MongoClient

MONGO_URI = "mongodb://localhost:27017"
MONGO_DB = "excavaite"
MONGO_COLL = "patent_data_523"
THIS_DIR = str(Path(__file__).parent.resolve())
BASE_DIR = str(Path(__file__).parent.parent.resolve())
INDEXES_DIR = "{}/indexes".format(BASE_DIR)

INDEX_NAME = "patent_data_529"

sys.path.append(BASE_DIR)

from core.indexes import USearchIndex
from core.vectorizers import SentBERTVectorizer

vectorizer = SentBERTVectorizer()
N_DIMS = vectorizer.embed("sample string").shape[0]  # 384

client = MongoClient(MONGO_URI)
collection = client[MONGO_DB][MONGO_COLL]
query = {"publicationDate": {"$gte": "20010807"}}

N = collection.count_documents(query)
cursor = collection.find(query)

labels = []
vectors = np.zeros((N, N_DIMS))
for i in tqdm(range(N)):
    if not cursor.alive:
        raise Exception("Mongo cursor terminated prematurely!")
    doc = cursor.next()
    dn = doc["publicationNumber"]
    text = []
    text.append(doc["title"])
    text.append(doc["abstract"])
    text.append(doc["description"])
    text.append(doc["claimInfo"])
    text = " ".join([t for t in text if t is not None])
    vector = vectorizer.embed(text)
    labels.append(dn)
    vectors[i] = vector

print("Saving vectors...")
np.save(f"{INDEXES_DIR}/{INDEX_NAME}_vectors.npy", vectors)

### EXISTING VECTORS ###
# import json

# vectors = np.load(f"{INDEXES_DIR}/{INDEX_NAME}_vectors.npy")
# with open(f"{INDEXES_DIR}/{INDEX_NAME}.items.json") as f:
#     labels = json.load(f)
### END EXISTING ###

print("Indexing...")
index = USearchIndex(name=INDEX_NAME)
index._index_dir = INDEXES_DIR
index.add_vectors(vectors, labels)
print("Index saved in {}".format(INDEXES_DIR))
