from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from supabase import create_client
import os
from dotenv import load_dotenv
import tempfile
import requests
import hashlib
import asyncio

# ✅ Load environment variables
load_dotenv()

# ✅ Environment values
google_api_key = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

# ✅ Initialize LLM, embeddings, vector store
model = ChatGroq(model="llama3-8b-8192", groq_api_key=groq_api_key)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
supabase = create_client(supabase_url, supabase_key)

# ✅ Generate unique hash from PDF bytes
def get_pdf_hash(pdf_bytes: bytes) -> str:
    return hashlib.md5(pdf_bytes).hexdigest()

# ✅ Embed document if not already in DB
def process_document_if_new(doc_url: str) -> str:
    response = requests.get(doc_url)
    if response.status_code != 200:
        raise ValueError("Failed to fetch document.")

    pdf_bytes = response.content
    doc_hash = get_pdf_hash(pdf_bytes)

    # ✅ Check for existing hash in DB
    vectorstore = SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name="documents",
        query_name="match_documents",
    )

    retriever = vectorstore.as_retriever(search_type="similarity", k=1, filter={"source": doc_hash})
    results = retriever.get_relevant_documents(doc_hash)

    if results:  # Already embedded
        return doc_hash

    # ✅ Save and embed document
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        tmp_file.write(pdf_bytes)
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=30)
    chunks = splitter.split_documents(documents)

    for doc in chunks:
        doc.metadata = {"source": doc_hash}

    SupabaseVectorStore.from_documents(
        chunks,
        embeddings,
        client=supabase,
        table_name="documents",
        query_name="match_documents",
        chunk_size=1000,
    )

    return doc_hash

# ✅ Query LLM using embedded document - Async
async def query_document_by_hash(questions: list[str], doc_hash: str) -> list[str]:
    vectorstore = SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name="documents",
        query_name="match_documents",
    )

    retriever = vectorstore.as_retriever(search_type="similarity", k=3, filter={"source": doc_hash})
    qa_chain = RetrievalQA.from_chain_type(llm=model, retriever=retriever, chain_type="stuff", return_source_documents=False)

    # ✅ Use asyncio.gather to run queries in parallel
    tasks = [qa_chain.ainvoke({"query": q}) for q in questions]
    results = await asyncio.gather(*tasks)

    # ✅ Extract answer only
    answers = [r["result"] for r in results]
    return answers