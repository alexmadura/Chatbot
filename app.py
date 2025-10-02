import streamlit as st
import ollama
import os
import subprocess
from pypdf import PdfReader
from langchain_community.chat_models import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

# --- Constants ---
CHROMA_PATH = "chroma_db"
SUPPORTED_EXTENSIONS = ['.pdf', '.py', '.java', 'js', '.txt', '.md']
LOGO_URL = "https://upload.wikimedia.org/wikipedia/commons/7/79/Siemens_Healthineers_logo.svg"

# --- Page Config ---
st.set_page_config(page_title="IPIPE Codelmair", layout="wide")

# --- Custom CSS for Branding ---
st.markdown("""
<style>
    /* Main background color for the app */
    .stApp {
        background-color: #1a1a1a; /* Dark Gray Background */
        color: #fafafa; /* Light text for the main app */
    }

    /* Sidebar styling - remains light */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        color: #1a1a1a; /* Text color for sidebar should be dark */
    }

    /* Ensure sidebar headers are dark */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p {
        color: #1a1a1a;
    }

    /* Button styling */
    .stButton>button {
        background-color: #cf4b00; /* Accessible Orange */
        color: white;
        border: none;
        border-radius: 4px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #cf4b00; /* Accessible Orange */
        opacity: 0.9;
        color: white;
    }

    /* Title and Header styling for the main area */
    h1, h2, h3 {
        color: #fafafa; /* Light Gray for text */
    }

    /* Chat message styling */
    [data-testid="stChatMessage"] {
        background-color: #2b2b2b; /* Slightly lighter than main background */
        border-radius: 8px;
    }

    /* Ensure alerts have readable text on dark background */
    [data-testid="stAlert"] {
        color: #1a1a1a;
    }
</style>
""", unsafe_allow_html=True)


# --- Helper Functions ---

def get_gpu_info():
    """Checks for NVIDIA GPU and returns its total memory in MiB."""
    try:
        # Run nvidia-smi command to get GPU memory
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
        # The command returns a string like "12288\n", so we strip and convert to int
        gpu_memory = int(result.strip())
        return gpu_memory
    except (FileNotFoundError, subprocess.CalledProcessError):
        # This will happen if nvidia-smi is not found (no NVIDIA GPU/drivers)
        # or if the command fails for other reasons.
        return None

def get_finetuning_recommendation(vram_mb):
    """Returns a recommendation based on the available VRAM in MiB."""
    if vram_mb is None:
        return (
            "**Keine NVIDIA-GPU gefunden.** Finetuning ist auf dieser Maschine "
            "nicht möglich, da das `nvidia-smi`-Kommando nicht gefunden wurde. "
            "Bitte stellen Sie sicher, dass die NVIDIA-Treiber korrekt installiert sind.",
            "error"
        )

    # Convert MiB to GiB for easier comparison
    vram_gb = vram_mb / 1024

    if vram_gb < 8:
        return (
            f"**GPU-Speicher: {vram_gb:.1f} GB.** Dieser Speicher ist für das "
            "Finetuning von modernen Sprachmodellen leider nicht ausreichend.",
            "error"
        )
    elif vram_gb < 16:
        return (
            f"**GPU-Speicher: {vram_gb:.1f} GB.** Mit diesem Speicher ist "
            "experimentelles Finetuning mit speziellen Techniken (wie LoRA/PEFT) "
            "möglich. Der Prozess kann jedoch langsam sein und erfordert eine "
            "sorgfältige Konfiguration.",
            "warning"
        )
    else: # vram_gb >= 16
        return (
            f"**GPU-Speicher: {vram_gb:.1f} GB.** Diese GPU ist gut für "
            "experimentelles Finetuning geeignet. Sie können mit Techniken "
            "wie LoRA/PEFT beginnen, um das Modell auf Ihren Daten zu trainieren.",
            "success"
        )

def check_ollama_status():
    try:
        ollama.list()
        return True
    except Exception:
        return False

def get_all_filepaths(folder_path):
    filepaths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                filepaths.append(os.path.join(root, file))
    return filepaths

def get_docs_from_filepaths(filepaths):
    docs = []
    for filepath in filepaths:
        try:
            content = ""
            if filepath.endswith('.pdf'):
                with open(filepath, 'rb') as f:
                    pdf_reader = PdfReader(f)
                    for page in pdf_reader.pages:
                        content += page.extract_text() or ""
            else:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            docs.append(Document(page_content=content, metadata={"source": filepath}))
        except Exception as e:
            st.error(f"Fehler beim Lesen der Datei {filepath}: {e}")
    return docs

def get_text_chunks_from_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(docs)

def add_to_chroma(chunks):
    # Initialize the Chroma client
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=OllamaEmbeddings(model="gemma:2b")
    )

    # Add documents to the store
    ids = [f"{chunk.metadata['source']}-{i}" for i, chunk in enumerate(chunks)]
    vector_store.add_documents(documents=chunks, ids=ids)
    vector_store.persist()

# --- RAG Chain Creation --- (No changes to these functions)
def get_vectorstore():
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=OllamaEmbeddings(model="gemma:2b")
    )

def get_context_retriever_chain(vector_store):
    llm = ChatOllama(model="gemma:2b")
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    return create_history_aware_retriever(llm, retriever, prompt)

def get_conversational_rag_chain(retriever_chain):
    llm = ChatOllama(model="gemma:2b")
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# --- App Layout ---
st.title("IPIPE Codelmair")

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "db_ready" not in st.session_state:
    st.session_state.db_ready = os.path.exists(CHROMA_PATH)

# Sidebar
with st.sidebar:
    st.image(LOGO_URL, use_column_width=True)
    st.header("Einstellungen")
    folder_path = st.text_input("Geben Sie den Pfad zu Ihrem Ordner ein:")

    if st.button("Ordner verarbeiten"):
        if folder_path and os.path.isdir(folder_path):
            with st.spinner(f"Durchsuche Ordner '{folder_path}'..."):
                filepaths = get_all_filepaths(folder_path)
                st.success(f"{len(filepaths)} unterstützte Dateien gefunden.")

            if filepaths:
                with st.spinner("Lese und verarbeite Dateien..."):
                    docs = get_docs_from_filepaths(filepaths)
                    chunks = get_text_chunks_from_docs(docs)
                with st.spinner("Vektor-Datenbank wird erstellt/aktualisiert..."):
                    add_to_chroma(chunks)
                st.session_state.db_ready = True
                st.success("Verarbeitung abgeschlossen!")
        else:
            st.error("Bitte geben Sie einen gültigen Ordnerpfad an.")

    st.info(
        """
        **Anleitung:**
        1. Kopieren Sie den vollständigen Pfad zu dem Ordner, den Sie analysieren möchten.
        2. Fügen Sie den Pfad in das Textfeld oben ein.
        3. Klicken Sie auf "Ordner verarbeiten".
        """
    )

    st.divider()

    st.subheader("Finetuning")
    if st.button("Hardware-Check für Finetuning"):
        with st.spinner("Überprüfe Hardware..."):
            vram_mb = get_gpu_info()
            recommendation, status = get_finetuning_recommendation(vram_mb)

            if status == "success":
                st.success(recommendation)
            elif status == "warning":
                st.warning(recommendation)
            else: # error
                st.error(recommendation)


# Main chat interface
st.header("Chat")

if not check_ollama_status():
    st.warning("**Ollama-Dienst nicht erreichbar!** Stelle sicher, dass Ollama läuft.")
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Stelle deine Frage hier..."):
    if not st.session_state.db_ready:
        st.warning("Bitte verarbeiten Sie zuerst einen Ordner.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            vector_store = get_vectorstore()
            retriever_chain = get_context_retriever_chain(vector_store)
            conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

            placeholder = st.empty()
            full_response = ""

            chat_history = [HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]) for msg in st.session_state.messages[:-1]]

            stream = conversation_rag_chain.stream({
                "chat_history": chat_history,
                "input": prompt
            })

            for chunk in stream:
                if "answer" in chunk and chunk["answer"] is not None:
                    full_response += chunk["answer"]
                    placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.rerun()