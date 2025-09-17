import streamlit as st
import ollama
from pypdf import PdfReader
from io import BytesIO
from langchain_community.chat_models import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- Helper Functions ---

@st.cache_data
def check_ollama_status():
    """Checks if the Ollama service is running. Cached to avoid re-checking on every interaction."""
    try:
        ollama.list()
        return True
    except Exception:
        return False

@st.cache_data
def get_text_from_files(files):
    """Extracts text from a list of uploaded files (PDFs and text-based)."""
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
def get_vectorstore_from_text(_text_hash, text):
    """Splits text, creates embeddings, and stores them in a FAISS vector store. Caches the resource."""
    if not text:
        return None
    with st.spinner("Vektor-Datenbank wird erstellt... Dieser Schritt wird nur einmal pro Datei-Set ausgeführt."):
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            text_chunks = text_splitter.split_text(text)

            embeddings = OllamaEmbeddings(model="gemma:2b")

            vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
            return vectorstore
        except Exception as e:
            st.error(f"Fehler beim Erstellen der Vektor-Datenbank: {e}")
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

# Sidebar
with st.sidebar:
    st.header("Einstellungen")
    uploaded_files = st.file_uploader(
        "1. Lade deine Dateien hoch",
        type=['pdf', 'py', 'java', 'js', 'txt', 'md'],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("2. Dateien verarbeiten"):
            raw_text = get_text_from_files(tuple(uploaded_files))
            if raw_text:
                st.session_state.vector_store = get_vectorstore_from_text(hash(raw_text), raw_text)
                st.success("Verarbeitung abgeschlossen!")

    st.info("**Wie es funktioniert:**\n1. Dateien hochladen.\n2. Auf 'Dateien verarbeiten' klicken.\n3. Fragen stellen.")
    st.warning("**Für bessere Ergebnisse:**\nStandardmäßig wird `gemma:2b` für Embeddings genutzt. Für höhere Genauigkeit `nomic-embed-text` via `ollama pull nomic-embed-text` installieren und im Code anpassen.")

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
        st.warning("Bitte lade zuerst Dateien hoch und klicke auf 'Dateien verarbeiten'.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
            conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

            placeholder = st.empty()
            full_response = ""

            # Convert session state messages to LangChain's format
            chat_history = [HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]) for msg in st.session_state.messages[:-1]] # Exclude the last user message

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
            # Rerun to show the last message in history
            st.rerun()
