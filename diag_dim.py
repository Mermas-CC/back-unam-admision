import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

models = ["models/embedding-001", "models/gemini-embedding-001"]

for model in models:
    try:
        result = genai.embed_content(model=model, content="Hola mundo")
        dim = len(result["embedding"])
        print(f"Model: {model} -> Dimension: {dim}")
    except Exception as e:
        print(f"Model: {model} -> Error: {e}")

import chromadb
PERSIST_DIR = "./chroma_db_v2"
if os.path.exists(PERSIST_DIR):
    db = chromadb.PersistentClient(path=PERSIST_DIR)
    try:
        col = db.get_collection("admision_unap")
        sample = col.get(limit=1, include=['embeddings'])
        if sample['embeddings']:
            print(f"DB Collection Dimension: {len(sample['embeddings'][0])}")
        else:
             print("DB Collection is empty or has no embeddings stored (unlikely if it was ingested correctly).")
    except Exception as e:
        print(f"DB Error: {e}")
