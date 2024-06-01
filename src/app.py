import os
import streamlit as st
from model import ChatModel
import rag_util

FILES_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "files")
)

st.title("LLM Chatbot RAG Asystent")

@st.cache_resource
def load_model():
    model = ChatModel(model_id="google/gemma-2b", device="cuda")
    return model

@st.cache_resource
def load_encoder():
    encoder = rag_util.Encoder(
        model_name="sentence-transformers/all-MiniLM-L12-v2", device="cpu"
    )
    return encoder

model = load_model()  # załaduj nasze modele raz, a następnie przechowaj je w pamięci podręcznej
encoder = load_encoder()

def save_file(uploaded_file):
    """funkcja pomocnicza do zapisywania dokumentów na dysku"""
    file_path = os.path.join(FILES_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

with st.sidebar:
    max_new_tokens = st.number_input("maksymalna liczba nowych tokenów", 128, 4096, 512)
    k = st.number_input("k", 1, 10, 3)
    uploaded_files = st.file_uploader(
        "Prześlij pliki PDF/docx jako kontekst", type=["PDF", "pdf", "doc", "docx"], accept_multiple_files=True
    )
    file_paths = []
    for uploaded_file in uploaded_files:
        file_paths.append(save_file(uploaded_file))
    if uploaded_files != []:
        docs = rag_util.load_and_split_pdfs(file_paths)
        DB = rag_util.FaissDb(docs=docs, embedding_function=encoder.embedding_function)

# Inicjalizacja historii czatu
if "messages" not in st.session_state:
    st.session_state.messages = []

# Wyświetlanie wiadomości czatu z historii po ponownym uruchomieniu aplikacji
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Akceptowanie danych wejściowych użytkownika
if prompt := st.chat_input("Zadaj mi pytanie!"):
    # Dodaj wiadomość użytkownika do historii czatu
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Wyświetl wiadomość użytkownika w kontenerze wiadomości czatu
    with st.chat_message("user"):
        st.markdown(prompt)

    # Wyświetl odpowiedź asystenta w kontenerze wiadomości czatu
    with st.chat_message("assistant"):
        user_prompt = st.session_state.messages[-1]["content"]
        context = (
            None if uploaded_files == [] else DB.similarity_search(user_prompt, k=k)
        )
        answer = model.generate(
            user_prompt, context=context, max_new_tokens=max_new_tokens
        )
        response = st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
