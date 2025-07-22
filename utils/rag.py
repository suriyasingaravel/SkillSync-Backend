import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set in .env")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=api_key)
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("resume_bullets", embedding_function=openai_ef)

def get_rag_examples(skills):
    results = {}
    for skill in skills:
        try:
            query = collection.query(query_texts=[skill], n_results=1)
            bullets = query['documents'][0] if query['documents'] else []
            if bullets:
                results[skill] = bullets[0]
        except Exception as e:
            results[skill] = f"(No example found: {e})"
    return results
