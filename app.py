"""
📚 Kitapyurdu Yorumlarıyla Kitap Öneri Asistanı
--------------------------------------------------
Bu proje, Hugging Face üzerindeki Kitapyurdu yorum verisetini
kullanarak RAG (Retrieval-Augmented Generation) temelli bir kitap
öneri chatbotu oluşturur.

Model: Gemini 2.0 Flash
Framework: Haystack
Vector DB: FAISS
Arayüz: Streamlit
"""

import os
import streamlit as st
from datasets import load_dataset
from haystack import Document, Pipeline
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.pipelines import Pipeline
from haystack.schema import Document
from dotenv import load_dotenv
import google.generativeai as genai

# --------------------------------------------------
# 🔧 1. Ortam Değişkenleri
# --------------------------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

if not GEMINI_API_KEY:
    st.error("⚠️ GEMINI_API_KEY bulunamadı. Lütfen .env dosyasına ekle.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# --------------------------------------------------
# 📦 2. Veri Seti (Kitapyurdu)
# --------------------------------------------------
@st.cache_data
def load_data():
    st.info("📥 Veri seti yükleniyor...")
    dataset = load_dataset("alibayram/kitapyurdu_yorumlar", split="train", token=HF_TOKEN)
    data = dataset.shuffle(seed=42).select(range(2000))  # küçük örnekle başla
    docs = []
    for item in data:
        content = f"Kitap: {item['book_name']}\nYazar: {item['author_name']}\nKategori: {item['category']}\nYorum: {item['comment']}"
        docs.append(Document(content=content, meta={"rating": item["rating"]}))
    return docs

docs = load_data()

# --------------------------------------------------
# 🧮 3. FAISS + Embedding Retriever
# --------------------------------------------------
@st.cache_resource
def init_vector_db():
    st.info("🔍 FAISS veritabanı oluşturuluyor...")
    document_store = FAISSDocumentStore(embedding_dim=384, faiss_index_factory_str="Flat")
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/paraphrase-MiniLM-L6-v2",
        model_format="sentence_transformers"
    )
     if not document_store.get_all_documents():
        document_store.write_documents(docs)
        document_store.update_embeddings(retriever)
    return document_store, retriever
document_store, retriever = init_vector_db()

# --------------------------------------------------
# 🧠 4. Gemini ile Cevaplama
# --------------------------------------------------
def generate_answer(context, query):
    prompt = f"""
Sen bir kitap öneri asistanısın. Aşağıdaki kullanıcı yorumlarından yararlanarak
soruya doğal, akıcı ve gerekçeli bir yanıt ver.

Kullanıcı Sorusu: {query}

İlgili Yorumlar:
{context}

Yanıt:
"""
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

# --------------------------------------------------
# 💬 5. Streamlit Arayüzü
# --------------------------------------------------
st.set_page_config(page_title="Kitap Öneri Asistanı", page_icon="📚")
st.title("📚 Kitapyurdu Yorumlarıyla Kitap Öneri Asistanı")
st.caption("RAG tabanlı bir kitap öneri chatbotu — powered by Gemini 2.0 Flash")

query = st.text_input("📖 Bir kitap türü, tema veya yazar hakkında soru sor:")
top_k = st.slider("Kaç benzer yorumu dikkate alalım?", 2, 10, 5)

if st.button("Öneri Getir"):
    if not query.strip():
        st.warning("Lütfen bir soru yazın.")
    else:
        with st.spinner("🔎 Yorumlar inceleniyor..."):
            retrieved_docs = retriever.retrieve(query, top_k=top_k)
            context = "\n\n".join([doc.content for doc in retrieved_docs])
        with st.spinner("💡 Gemini yanıt üretiyor..."):
            answer = generate_answer(context, query)
        st.subheader("✨ Önerilen Kitaplar / Yanıt")
        st.write(answer)
        with st.expander("📚 Kullanılan Yorumlar"):
            for doc in retrieved_docs:
                st.markdown(f"- {doc.content[:200]}...")

st.sidebar.info("Developed by Nihal Metin — Akbank GenAI Bootcamp")
