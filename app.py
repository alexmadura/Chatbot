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
LLAVA_MODEL = "llava" # Model for image description
EMBEDDING_MODEL = "gemma:2b" # Model for text embeddings

# --- Helper Functions ---

def file_list_hasher(files):
    if not files:
        return ""
    return "_".join([f"{file.name}_{file.size}" for file in files])

def get_image_description(image_data):
    """Generates a description for an image using the LLaVA model."""
    try:
        res = ollama.chat(
            model=LLAVA_MODEL,
            messages=[
                {
                    'role': 'user',
                    'content': 'Beschreibe dieses Bild detailliert.',
                    'images': [image_data]
                }
            ]
        )
        return res['message']['content']
    except Exception as e:
        st.error(f"Fehler bei der Bildbeschreibung: {e}")
        return ""

@st.cache_data
def check_ollama_status():
    try:
        ollama.list()
        return True
    except Exception:
        return False

@st.cache_data(hash_funcs={list: file_list_hasher})
def get_text_and_images_from_files(files):
    """Extracts text from PDFs/code and generates descriptions for images."""
    raw_text = ""
    processed_images = []

    for file in files:
        file_extension = file.name.split('.')[-1].lower()
        if file_extension in ['png', 'jpg', 'jpeg']:
            image_bytes = file.getvalue()
            description = get_image_description(image_bytes)
            raw_text += f"\nBildbeschreibung für {file.name}:\n{description}\n"
            # We keep the raw bytes to display later
            processed_images.append({'name': file.name, 'bytes': image_bytes})
        elif file_extension == 'pdf':
            try:
                pdf_bytes = BytesIO(file.getvalue())
                pdf_reader = PdfReader(pdf_bytes)
                for page in pdf_reader.pages:
                    raw_text += page.extract_text() or ""
            except Exception as e:
                st.error(f"Fehler beim Lesen der PDF-Datei {file.name}: {e}")
        else:
            try:
                raw_text += file.getvalue().decode('utf-8', errors='ignore')
            except Exception as e:
                st.error(f"Fehler beim Lesen der Text-Datei {file.name}: {e}")

    return raw_text, processed_images

@st.cache_resource
def get_vectorstore_from_text(text, store_name):
    if not text or not store_name:
        return None

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
            vectorstore = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
            return vectorstore
        except Exception as e:
            st.error(f"Fehler beim Laden der Vektor-Datenbank: {e}")
            return None

# --- RAG Chain & Multimodal Chat ---

def get_conversational_rag_chain(vector_store):
    llm = ChatOllama(model=LLAVA_MODEL)
    retriever = vector_store.as_retriever()

    # History-aware retriever chain
    retriever_prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    history_retriever_chain = create_history_aware_retriever(ChatOllama(model=EMBEDDING_MODEL), retriever, retriever_prompt)

    # Conversational RAG chain
    rag_prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context. If an image was part of the user's prompt, it is included in the input.\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, rag_prompt)
    return create_retrieval_chain(history_retriever_chain, stuff_documents_chain)

# --- App Layout ---
st.set_page_config(page_title="Multimodaler Chatbot", layout="wide")
st.title("📄 Chatte mit deinen Dokumenten und Bildern")

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_images" not in st.session_state:
    st.session_state.processed_images = []
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
            "1. Lade deine Dateien hoch (inkl. Bilder)",
            type=['pdf', 'py', 'java', 'js', 'txt', 'md', 'png', 'jpg', 'jpeg'],
            accept_multiple_files=True
        )
        db_name = st.text_input("2. Name für die Wissensdatenbank:")

        if st.button("3. Verarbeiten und Speichern"):
            if uploaded_files and db_name:
                raw_text, processed_images = get_text_and_images_from_files(uploaded_files)
                st.session_state.processed_images = processed_images
                if raw_text:
                    st.session_state.vector_store = get_vectorstore_from_text(raw_text, db_name)
                    if st.session_state.vector_store:
                        st.success(f"Wissensdatenbank '{db_name}' erstellt!")
            else:
                st.warning("Bitte Dateien hochladen UND einen Namen eingeben.")

    elif mode == "Bestehende Wissensdatenbank laden":
        saved_dbs = [d for d in os.listdir(VECTOR_STORE_DIR) if os.path.isdir(os.path.join(VECTOR_STORE_DIR, d))]
        if saved_dbs:
            selected_db = st.selectbox("Wähle eine Wissensdatenbank:", saved_dbs)
            if st.button("Laden"):
                st.session_state.vector_store = load_vectorstore(selected_db)
                if st.session_state.vector_store:
                    st.success(f"Wissensdatenbank '{selected_db}' geladen!")
        else:
            st.info("Keine Wissensdatenbanken gefunden.")

    st.info(f"**Hinweis:** Für Bilderkennung wird `{LLAVA_MODEL}` benötigt. (`ollama pull {LLAVA_MODEL}`)")

# Main chat interface
st.header("Chat")

if not check_ollama_status():
    st.warning("**Ollama-Dienst nicht erreichbar!**")
    st.stop()

# Display chat messages from history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Render images if they exist in the message
        if "images" in msg:
            for img_bytes in msg["images"]:
                st.image(img_bytes)
        st.markdown(msg["content"])

# Chat input
chat_input_container = st.container()
with chat_input_container:
    prompt = st.chat_input("Stelle deine Frage hier...")
    image_input = st.file_uploader("Bild hinzufügen:", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

if prompt:
    if st.session_state.vector_store is None:
        st.warning("Bitte erstelle oder lade zuerst eine Wissensdatenbank.")
    else:
        user_images = [img.getvalue() for img in image_input] if image_input else []

        # Add user message to state
        st.session_state.messages.append({"role": "user", "content": prompt, "images": user_images})

        with st.chat_message("user"):
            if user_images:
                for img_bytes in user_images:
                    st.image(img_bytes)
            st.markdown(prompt)

        with st.chat_message("assistant"):
            conversation_rag_chain = get_conversational_rag_chain(st.session_state.vector_store)

            placeholder = st.empty()
            full_response = ""

            # Prepare chat history for the model
            chat_history = []
            for msg in st.session_state.messages[:-1]:
                role = msg["role"]
                content = msg["content"]
                if role == "user":
                    chat_history.append(HumanMessage(content=content))
                else:
                    chat_history.append(AIMessage(content=content))

            # Include image data in the final prompt if present
            final_input = {"input": prompt, "chat_history": chat_history}
            if user_images:
                final_input["images"] = user_images

            stream = conversation_rag_chain.stream(final_input)

            for chunk in stream:
                if "answer" in chunk and chunk["answer"] is not None:
                    full_response += chunk["answer"]
                    placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.rerun()
