import os
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

load_dotenv()

# ENV vars
API_KEY      = os.getenv("PINECONE_API_KEY")
ENV          = os.getenv("PINECONE_ENV")           # e.g. "us-east1-aws"
INDEX_NAME   = os.getenv("PINECONE_INDEX_NAME")
EMB_MODEL    = os.getenv("EMBEDDING_MODEL_NAME")

# Pinecone client & index
pc    = Pinecone(api_key=API_KEY, environment=ENV)
index = pc.Index(INDEX_NAME)

# Local embedder
embedder = SentenceTransformer(EMB_MODEL)

def chunk_text(text, max_chars=2000):
    if len(text) <= max_chars:
        return [text]
    paras, curr, chunks = text.split("\n"), "", []
    for p in paras:
        if len(curr) + len(p) > max_chars:
            chunks.append(curr); curr = p
        else:
            curr += ("\n" + p)
    if curr:
        chunks.append(curr)
    return chunks

def main():
    with open("data/lifeafter_faq.json", encoding="utf-8") as f:
        faqs = json.load(f)

    for faq in faqs:
        passages = chunk_text(faq["content"])
        vectors = []
        for i, txt in enumerate(passages):
            raw = embedder.encode(txt)
            emb = raw.tolist() if hasattr(raw, "tolist") else list(raw)
            meta = {"id": faq["id"], "title": faq["title"], "text": txt}
            vectors.append((f"{faq['id']}-{i}", emb, meta))

        index.upsert(vectors=vectors)
        print(f"Upserted {len(passages)} chunks for '{faq['id']}'")

if __name__ == "__main__":
    main()
