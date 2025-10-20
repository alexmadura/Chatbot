import streamlit as st
import ollama
import os
from pypdf import PdfReader
from io import BytesIO
from langchain_community.chat_models import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- App Configuration ---
VECTOR_STORE_DIR = "vector_stores"

# --- Helper Functions ---

def file_list_hasher(files):
    """Create a hash from a list of file-like objects."""
    if not files:
        return ""
    return "_".join([f"{file.name}_{file.size}" for file in files])

@st.cache_data
def check_ollama_status():
    """Checks if the Ollama service is running."""
    try:
        ollama.list()
        return True
    except Exception:
        return False

@st.cache_data(hash_funcs={list: file_list_hasher})
def get_text_from_files(files):
    """Extracts text from a list of uploaded files."""
    raw_text = ""
    for file in files:
        file_extension = file.name.split('.')[-1].lower()
        try:
            if file_extension == 'pdf':
                pdf_bytes = BytesIO(file.getvalue())
                pdf_reader = PdfReader(pdf_bytes)
                for page in pdf_reader.pages:
                    raw_text += page.extract_text() or ""
            else:
                raw_text += file.getvalue().decode('utf-8', errors='ignore')
        except Exception as e:
            st.error(f"Fehler beim Lesen der Datei {file.name}: {e}")
    return raw_text

@st.cache_resource
def get_vectorstore_from_text(text, store_name):
    """Creates and saves a FAISS vector store from text."""
    if not text or not store_name:
        return None

    store_path = os.path.join(VECTOR_STORE_DIR, store_name)
    if os.path.exists(store_path):
        st.warning(f"Wissensdatenbank '{store_name}' existiert bereits. Bitte wählen Sie einen anderen Namen.")
        return None

    with st.spinner("Vektor-Datenbank wird erstellt und gespeichert..."):
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            text_chunks = text_splitter.split_text(text)

            embeddings = OllamaEmbeddings(model="gemma:2b")

            vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
            vectorstore.save_local(store_path)
            return vectorstore
        except Exception as e:
            st.error(f"Fehler beim Erstellen der Vektor-Datenbank: {e}")
            return None

@st.cache_resource
def load_vectorstore(store_name):
    """Loads a FAISS vector store from the local disk."""
    store_path = os.path.join(VECTOR_STORE_DIR, store_name)
    if not os.path.exists(store_path):
        st.error(f"Wissensdatenbank '{store_name}' nicht gefunden.")
        return None

    with st.spinner(f"Lade Wissensdatenbank '{store_name}'..."):
        try:
            embeddings = OllamaEmbeddings(model="gemma:2b")
            vectorstore = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
            return vectorstore
        except Exception as e:
            st.error(f"Fehler beim Laden der Vektor-Datenbank: {e}")
            return None

# --- RAG Chain Creation ---

def get_context_retriever_chain(vector_store):
    llm = ChatOllama(model="gemma:2b")
    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

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
st.set_page_config(page_title="Lokaler Chatbot mit Gemma", layout="wide")
st.title("📄 Chatte mit deinen Dokumenten und deinem Code")

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if not os.path.exists(VECTOR_STORE_DIR):
    os.makedirs(VECTOR_STORE_DIR)

# Sidebar
with st.sidebar:
    st.header("Einstellungen")

    mode = st.radio(
        "Wählen Sie einen Modus:",
        ("Neue Wissensdatenbank erstellen", "Bestehende Wissensdatenbank laden")
    )

    if mode == "Neue Wissensdatenbank erstellen":
        uploaded_files = st.file_uploader(
            "1. Lade deine Dateien hoch",
            type=['pdf', 'py', 'java', 'js', 'txt', 'md'],
            accept_multiple_files=True
        )
        db_name = st.text_input("2. Gib einen Namen für die neue Wissensdatenbank ein:")

        if st.button("3. Dateien verarbeiten und speichern"):
            if uploaded_files and db_name:
                raw_text = get_text_from_files(uploaded_files)
                if raw_text:
                    st.session_state.vector_store = get_vectorstore_from_text(raw_text, db_name)
                    if st.session_state.vector_store:
                        st.success(f"Wissensdatenbank '{db_name}' erfolgreich erstellt und gespeichert!")
            else:
                st.warning("Bitte lade Dateien hoch UND gib einen Namen ein.")

    elif mode == "Bestehende Wissensdatenbank laden":
        saved_dbs = [d for d in os.listdir(VECTOR_STORE_DIR) if os.path.isdir(os.path.join(VECTOR_STORE_DIR, d))]
        if saved_dbs:
            selected_db = st.selectbox("Wähle eine Wissensdatenbank:", saved_dbs)
            if st.button("Laden"):
                st.session_state.vector_store = load_vectorstore(selected_db)
                if st.session_state.vector_store:
                    st.success(f"Wissensdatenbank '{selected_db}' geladen!")
        else:
            st.info("Noch keine Wissensdatenbanken gespeichert.")

# Main chat interface
st.header("Chat")

if not check_ollama_status():
    st.warning("**Ollama-Dienst nicht erreichbar!** Stelle sicher, dass Ollama läuft.")
    st.stop()

# Display chat messages from history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Stelle deine Frage hier..."):
    if st.session_state.vector_store is None:
        st.warning("Bitte erstelle oder lade zuerst eine Wissensdatenbank in der Seitenleiste.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
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
