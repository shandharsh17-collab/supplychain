import streamlit as st
from PyPDF2 import PdfReader
from google import genai  # Gemini v2 SDK

# -------------------------------
# CONFIGURE GEMINI CLIENT
# -------------------------------
client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

# -------------------------------
# PDF TEXT EXTRACTION
# -------------------------------
def extract_text_from_pdf(uploaded_file):
    pdf = PdfReader(uploaded_file)
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# -------------------------------
# SPLIT TEXT INTO CHUNKS
# -------------------------------
def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# -------------------------------
# RETRIEVE RELEVANT CHUNKS (KEYWORD MATCH)
# -------------------------------
def retrieve_relevant_chunks(query, chunks, top_k=3):
    query = query.lower()
    scores = []
    for chunk in chunks:
        score = sum([1 for word in query.split() if word in chunk.lower()])
        scores.append(score)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [chunks[i] for i in top_indices]

# -------------------------------
# ASK GEMINI LLM
# -------------------------------
def ask_gemini(query, context_chunks):
    prompt = f"""
You are a supply chain risk analysis assistant.
Use the following supplier/tariff information to answer the question.

Context:
{' '.join(context_chunks)}

Question: {query}
Answer:
"""
    # Gemini v2 call
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return response.text

# -------------------------------
# STREAMLIT APP
# -------------------------------
st.title("📦 Supply Chain Risk Analysis Chatbot (Gemini v2)")

uploaded_file = st.file_uploader("Upload Supplier/Tariff Document (PDF)", type=["pdf"])

if uploaded_file:
    st.success("PDF uploaded successfully ✅")
    with st.spinner("Extracting and chunking document..."):
        text = extract_text_from_pdf(uploaded_file)
        chunks = split_text(text)
    st.success(f"Document processed into {len(chunks)} chunks!")

    query = st.text_input("Ask a supply chain risk question:")
    if query:
        relevant_chunks = retrieve_relevant_chunks(query, chunks)
        answer = ask_gemini(query, relevant_chunks)
        st.markdown("### 💡 Answer:")
        st.write(answer)


