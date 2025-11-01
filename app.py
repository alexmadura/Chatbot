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
from PIL import Image

# --- App Configuration ---
VECTOR_STORE_DIR = "vector_stores"
EMBEDDING_MODEL = "gemma:2b"

# --- Helper Functions ---

def file_list_hasher(files):
    if not files: return ""
    return "_".join([f"{file.name}_{file.size}" for file in files])

def get_image_description(image_data, model_name):
    try:
        res = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': 'Beschreibe dieses Bild detailliert.', 'images': [image_data]}]
        )
        return res['message']['content']
    except Exception as e:
        st.error(f"Fehler bei der Bildbeschreibung mit '{model_name}': {e}")
        return ""

@st.cache_data
def check_ollama_status():
    try:
        ollama.list()
        return True
    except Exception:
        return False

@st.cache_data(hash_funcs={list: file_list_hasher})
def get_text_and_images_from_files(files, model_name):
    raw_text = ""
    for file in files:
        file_extension = file.name.split('.')[-1].lower()
        if file_extension in ['png', 'jpg', 'jpeg']:
            image_bytes = file.getvalue()
            description = get_image_description(image_bytes, model_name)
            raw_text += f"\nBildbeschreibung für {file.name}:\n{description}\n"
        elif file_extension == 'pdf':
            try:
                pdf_bytes = BytesIO(file.getvalue())
                pdf_reader = PdfReader(pdf_bytes)
                for page in pdf_reader.pages:
                    raw_text += page.extract_text() or ""
            except Exception as e:
                st.error(f"Fehler beim Lesen der PDF {file.name}: {e}")
        else:
            try:
                raw_text += file.getvalue().decode('utf-8', errors='ignore')
            except Exception as e:
                st.error(f"Fehler beim Lesen der Datei {file.name}: {e}")
    return raw_text

@st.cache_resource
def get_vectorstore_from_text(text, store_name):
    if not text or not store_name: return None
    store_path = os.path.join(VECTOR_STORE_DIR, store_name)
    if os.path.exists(store_path):
        st.warning(f"Wissensdatenbank '{store_name}' existiert bereits.")
        return None
    with st.spinner("Vektor-Datenbank wird erstellt..."):
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            text_chunks = text_splitter.split_text(text)
            embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
            vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
            vectorstore.save_local(store_path)
            return vectorstore
        except Exception as e:
            st.error(f"Fehler beim Erstellen der Vektor-Datenbank: {e}")
            return None

@st.cache_resource
def load_vectorstore(store_name):
    store_path = os.path.join(VECTOR_STORE_DIR, store_name)
    if not os.path.exists(store_path):
        st.error(f"Wissensdatenbank '{store_name}' nicht gefunden.")
        return None
    with st.spinner(f"Lade Wissensdatenbank '{store_name}'..."):
        try:
            embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
            return FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            st.error(f"Fehler beim Laden der Vektor-Datenbank: {e}")
            return None

def get_conversational_rag_chain(vector_store, model_name):
    llm = ChatOllama(model=model_name)
    retriever = vector_store.as_retriever()
    retriever_prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    history_retriever_chain = create_history_aware_retriever(ChatOllama(model=EMBEDDING_MODEL), retriever, retriever_prompt)
    rag_prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, rag_prompt)
    return create_retrieval_chain(history_retriever_chain, stuff_documents_chain)

# --- App Layout ---
st.set_page_config(page_title="Multimodaler Chatbot", layout="wide")
st.title("📄 Chatte mit deinen Dokumenten und Bildern")

# Session state initialization
if "messages" not in st.session_state: st.session_state.messages = []
if "vector_store" not in st.session_state: st.session_state.vector_store = None
if "selected_model" not in st.session_state: st.session_state.selected_model = "llava"
if not os.path.exists(VECTOR_STORE_DIR): os.makedirs(VECTOR_STORE_DIR)

# Sidebar
with st.sidebar:
    st.header("Einstellungen")
    st.session_state.selected_model = st.selectbox(
        "Wählen Sie das multimodale Modell:",
        ("llava", "paligemma")
    )
    mode = st.radio("Modus:", ("Neue Wissensdatenbank", "Bestehende laden"))
    if mode == "Neue Wissensdatenbank":
        uploaded_files = st.file_uploader("1. Dateien hochladen", type=['pdf', 'py', 'java', 'js', 'txt', 'md', 'png', 'jpg', 'jpeg'], accept_multiple_files=True)
        db_name = st.text_input("2. Name für Wissensdatenbank:")
        if st.button("3. Verarbeiten & Speichern"):
            if uploaded_files and db_name:
                raw_text = get_text_and_images_from_files(uploaded_files, st.session_state.selected_model)
                if raw_text:
                    st.session_state.vector_store = get_vectorstore_from_text(raw_text, db_name)
                    if st.session_state.vector_store: st.success(f"Wissensdatenbank '{db_name}' erstellt!")
            else:
                st.warning("Bitte Dateien hochladen UND einen Namen eingeben.")
    else:
        saved_dbs = [d for d in os.listdir(VECTOR_STORE_DIR) if os.path.isdir(os.path.join(VECTOR_STORE_DIR, d))]
        if saved_dbs:
            selected_db = st.selectbox("Wähle eine Wissensdatenbank:", saved_dbs)
            if st.button("Laden"):
                st.session_state.vector_store = load_vectorstore(selected_db)
                if st.session_state.vector_store: st.success(f"Wissensdatenbank '{selected_db}' geladen!")
        else:
            st.info("Keine Wissensdatenbanken gefunden.")
    st.info("Multimodale Modelle wie `llava` oder `paligemma` müssen via Ollama installiert sein.")

# Main chat interface
st.header("Chat")
if not check_ollama_status():
    st.warning("**Ollama-Dienst nicht erreichbar!**")
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if "images" in msg and msg["images"]:
            for img_bytes in msg["images"]: st.image(img_bytes)
        st.markdown(msg["content"])

prompt = st.chat_input("Stelle deine Frage hier...")
image_input = st.file_uploader("Bild hinzufügen:", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

if prompt:
    user_images = [img.getvalue() for img in image_input] if image_input else []
    st.session_state.messages.append({"role": "user", "content": prompt, "images": user_images})

    with st.chat_message("user"):
        if user_images:
            for img_bytes in user_images: st.image(img_bytes)
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        # If there are images, call the model directly (no RAG)
        if user_images:
            try:
                res = ollama.chat(
                    model=st.session_state.selected_model,
                    messages=[{'role': 'user', 'content': prompt, 'images': user_images}]
                )
                full_response = res['message']['content']
            except Exception as e:
                full_response = f"Fehler bei der Verarbeitung des Bildes: {e}"
        # If no images, use the RAG chain with the knowledge base
        elif st.session_state.vector_store is not None:
            conversation_rag_chain = get_conversational_rag_chain(st.session_state.vector_store, st.session_state.selected_model)
            chat_history = [HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]) for msg in st.session_state.messages[:-1]]
            try:
                stream = conversation_rag_chain.stream({"chat_history": chat_history, "input": prompt})
                for chunk in stream:
                    if "answer" in chunk and chunk["answer"] is not None:
                        full_response += chunk["answer"]
                        placeholder.markdown(full_response + "▌")
            except Exception as e:
                full_response = f"Fehler bei der Ausführung der RAG-Chain: {e}"
        else:
            full_response = "Bitte erstelle oder lade zuerst eine Wissensdatenbank."

        placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.rerun()
