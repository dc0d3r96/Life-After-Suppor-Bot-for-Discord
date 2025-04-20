import os
import json
from dotenv import load_dotenv
import discord
from discord.ext import commands
import Pinecone
from sentence_transformers import SentenceTransformer
from huggingface_hub.inference_api import InferenceApi

# ────────────────────
# 1) Ortam değişkenleri
load_dotenv()
DISCORD_TOKEN        = os.getenv("DISCORD_TOKEN")
PINECONE_API_KEY     = os.getenv("PINECONE_API_KEY")
PINECONE_ENV         = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME  = os.getenv("PINECONE_INDEX_NAME")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
HF_TOKEN             = os.getenv("HF_TOKEN")
HF_MODEL_NAME        = os.getenv("HF_MODEL_NAME")

# ────────────────────
# 2) JSON dosyasını yükle (lokal yedek veri)
with open("data/lifeafter_faq.json", encoding="utf-8") as f:
    FAQS = json.load(f)

# ────────────────────
# 3) Pinecone ve embedder hazırlığı
pc       = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index    = pc.Index(PINECONE_INDEX_NAME)
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
hf_api   = InferenceApi(repo_id=HF_MODEL_NAME, token=HF_TOKEN)

# ────────────────────
# 4) Discord bot
intents = discord.Intents.default()
bot     = commands.Bot(command_prefix=None, intents=intents)
tree    = bot.tree

RETRIEVAL_THRESHOLD = 0.6  # Deneme için biraz indirdik

def simple_local_search(question: str) -> str | None:
    q = question.lower()
    for faq in FAQS:
        title = faq.get("title","").lower()
        content = faq.get("content","").lower()
        # başlıkta sorunun kelimeleri varsa
        if all(tok in title for tok in q.split()):
            return faq["content"]
        # içerikte sorunun baş parçası varsa
        if q in content:
            return faq["content"]
    return None

async def generate_answer(question: str) -> str:
    # A) Lokal JSON araması
    local = simple_local_search(question)
    if local:
        return local

    # B) Pinecone’dan semantik sorgu
    raw   = embedder.encode(question)
    q_emb = raw.tolist() if hasattr(raw, "tolist") else list(raw)
    res   = index.query(vector=q_emb, top_k=5, include_metadata=True)
    matches = res.get("matches", [])
    if matches and matches[0].get("score",0) >= RETRIEVAL_THRESHOLD:
        return matches[0]["metadata"]["text"].strip()

    # C) LLM’e devret
    context = "\n".join(f"- {m['metadata']['text']}" for m in matches)
    prompt = (
        "Life After Nights Falls.\n"
        "Aşağıdaki kaynak metinleri ve kendi deneyimini kullanarak soruya açık, "
        "adım adım ve gereksiz tekrar içermeyen net bir yanıt ver.\n\n"
        f"### Kaynak Metinler (varsa):\n{context}\n\n"
        f"### Soru:\n{question}\n\n"
        "### Cevap:"
    )
    gen = hf_api(inputs=prompt, params={"max_new_tokens":200,"temperature":0.3})
    if isinstance(gen, dict) and "generated_text" in gen:
        return gen["generated_text"].strip()
    if isinstance(gen, list) and gen and isinstance(gen[0], dict):
        return gen[0].get("generated_text","").strip()
    return str(gen).strip()

@tree.command(name="sor", description="Photoshop sorularınızı yanıtlar")
async def slash_sor(interaction: discord.Interaction, soru: str):
    await interaction.response.defer()
    try:
        answer = await generate_answer(soru)
        await interaction.followup.send(answer)
    except Exception as e:
        await interaction.followup.send(f"Üzgünüm, bir hata oluştu: {e}")

@bot.event
async def on_ready():
    await tree.sync()
    print(f"{bot.user} giriş yaptı, slash komutları senkronize edildi.")

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
