import requests
import os
import io
import base64
import PIL
from PIL import Image, ImageEnhance
import numpy as np
import streamlit as st
import cohere
from google import genai
from google.genai import types
import fitz  # PyMuPDF
from dotenv import load_dotenv

# --- Load API Keys from .env ---
load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY")
google_api_key = os.getenv("GEMINI_API_KEY")

# --- Streamlit App Configuration ---
st.set_page_config(layout="wide", page_title="AI Vision RAG Explorer")

# --- Custom CSS for UI Styling ---
st.markdown("""
    <style>
        /* Background Gradient */
        .stApp {
            background: linear-gradient(145deg, #0f2027, #203a43, #2c5364);
            color: white;
        }
        /* Title */
        h1 {
            font-size: 2.8rem !important;
            background: -webkit-linear-gradient(45deg, #4CAF50, #00C9FF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            font-weight: bold;
        }
        /* Sidebar Upload */
        .css-1d391kg {
            background: #1c1c1e;
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.4);
        }
        /* Buttons */
        .stButton>button {
            background: linear-gradient(90deg, #4CAF50, #00C9FF);
            color: white;
            border-radius: 10px;
            font-size: 16px;
            padding: 0.6rem 1.2rem;
            border: none;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #00C9FF, #4CAF50);
            transform: scale(1.05);
        }
        /* Image Card */
        .image-card {
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.6);
            margin-bottom: 20px;
        }
        /* Answer Bubble */
        .answer-box {
            background: rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 15px;
            margin-top: 15px;
            font-size: 1.1rem;
        }
        /* Footer */
        .footer {
            text-align: center;
            font-size: 14px;
            color: #aaa;
            margin-top: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Enhanced Title ---
st.markdown("<h1> AI Vision RAG Explorer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;font-size:18px;color:#ccc;'>Unlock insights from documents & images with Cohere + Gemini 🚀</p>", unsafe_allow_html=True)

# --- Initialize API Clients ---
co, genai_client = None, None
if cohere_api_key and google_api_key:
    try:
        co = cohere.ClientV2(api_key=cohere_api_key)
    except Exception as e:
        st.error(f"⚠️ Cohere Init Failed: {e}")
    try:
        genai_client = genai.Client(api_key=google_api_key, http_options=types.HttpOptions(api_version="v1"))
    except Exception as e:
        st.error(f"⚠️ Gemini Init Failed: {e}")
else:
    st.warning("⚠️ API Keys missing. Please add them in your `.env` file.")

# --- Helper Functions ---
max_pixels = 1568 * 1568

def enhance_image(pil_image: PIL.Image.Image) -> PIL.Image.Image:
    """Auto-enhance uploaded images for better readability."""
    enhancer_contrast = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer_contrast.enhance(1.2)
    enhancer_brightness = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer_brightness.enhance(1.1)
    return pil_image

def resize_image(pil_image: PIL.Image.Image) -> None:
    org_width, org_height = pil_image.size
    if org_width * org_height > max_pixels:
        scale_factor = (max_pixels / (org_width * org_height)) ** 0.5
        new_width = int(org_width * scale_factor)
        new_height = int(org_height * scale_factor)
        pil_image.thumbnail((new_width, new_height))

def base64_from_image(img_path: str) -> str:
    pil_image = PIL.Image.open(img_path)
    pil_image = enhance_image(pil_image)
    img_format = pil_image.format if pil_image.format else "PNG"
    resize_image(pil_image)
    with io.BytesIO() as buf:
        pil_image.save(buf, format=img_format)
        buf.seek(0)
        return f"data:image/{img_format.lower()};base64," + base64.b64encode(buf.read()).decode("utf-8")

def pil_to_base64(pil_image: PIL.Image.Image) -> str:
    pil_image = enhance_image(pil_image)
    img_format = pil_image.format if pil_image.format else "PNG"
    resize_image(pil_image)
    with io.BytesIO() as buf:
        pil_image.save(buf, format=img_format)
        buf.seek(0)
        return f"data:image/{img_format.lower()};base64," + base64.b64encode(buf.read()).decode("utf-8")

@st.cache_data(ttl=3600, show_spinner=False)
def compute_image_embedding(base64_img: str, _cohere_client) -> np.ndarray | None:
    try:
        api_response = _cohere_client.embed(
            model="embed-v4.0",
            input_type="search_document",
            embedding_types=["float"],
            images=[base64_img],
        )
        if api_response.embeddings and api_response.embeddings.float:
            return np.asarray(api_response.embeddings.float[0])
        return None
    except Exception as e:
        st.error(f"⚠️ Embedding error: {e}")
        return None

# --- PDF Processing ---
def process_pdf_file(pdf_file, cohere_client, base_output_folder="pdf_pages"):
    page_image_paths, page_embeddings = [], []
    pdf_filename = pdf_file.name
    output_folder = os.path.join(base_output_folder, os.path.splitext(pdf_filename)[0])
    os.makedirs(output_folder, exist_ok=True)

    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        progress = st.progress(0.0)
        for i, page in enumerate(doc.pages()):
            page_num = i + 1
            img_path = os.path.join(output_folder, f"page_{page_num}.png")
            pix = page.get_pixmap(dpi=150)
            pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pil_image = enhance_image(pil_image)  # Auto enhance
            pil_image.save(img_path, "PNG")

            base64_img = pil_to_base64(pil_image)
            emb = compute_image_embedding(base64_img, cohere_client)
            if emb is not None:
                page_embeddings.append(emb)
                page_image_paths.append(img_path)

            progress.progress((i+1)/len(doc))
        doc.close()
        progress.empty()
        return page_image_paths, page_embeddings
    except Exception as e:
        st.error(f"⚠️ PDF processing failed: {e}")
        return [], []

# --- Search & Answer ---
def search(question, co_client, embeddings, image_paths):
    try:
        resp = co_client.embed(
            model="embed-v4.0",
            input_type="search_query",
            embedding_types=["float"],
            texts=[question],
        )
        query_emb = np.asarray(resp.embeddings.float[0])
        sims = np.dot(query_emb, embeddings.T)
        idx = np.argmax(sims)
        return image_paths[idx]
    except Exception as e:
        st.error(f"⚠️ Search error: {e}")
        return None

def answer(question, img_path, gemini_client):
    try:
        with open(img_path, "rb") as f:
            image_bytes = f.read()
        mime = "image/png" if img_path.endswith(".png") else "image/jpeg"
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime)

        prompt = f"Answer this question based on the image:\n\nQuestion: {question}"
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, image_part]
        )
        return response.text
    except Exception as e:
        st.error(f"⚠️ Answer error: {e}")
        return f"Failed to generate answer: {e}"

# --- Session State ---
if "image_paths" not in st.session_state:
    st.session_state.image_paths = []
if "doc_embeddings" not in st.session_state:
    st.session_state.doc_embeddings = None

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Drag & Drop Files Here",
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

if uploaded_files and co:
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            paths, embs = process_pdf_file(uploaded_file, co)
        else:
            img_path = os.path.join("uploaded", uploaded_file.name)
            os.makedirs("uploaded", exist_ok=True)
            with open(img_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            base64_img = base64_from_image(img_path)
            emb = compute_image_embedding(base64_img, co)
            paths, embs = [img_path], [emb]
        if paths and embs:
            st.session_state.image_paths.extend(paths)
            if st.session_state.doc_embeddings is None:
                st.session_state.doc_embeddings = np.vstack(embs)
            else:
                st.session_state.doc_embeddings = np.vstack((st.session_state.doc_embeddings, np.vstack(embs)))
    st.sidebar.success("✅ Files Uploaded!")

# --- Ask Question ---
st.markdown(
    "<h3 style='text-align: center; color: #ccc;'> Query your files and retrieve AI powered insights.</h3>",
    unsafe_allow_html=True
)
# st.markdown(" ## Query your files and retrieve AI-powered insights.")
# st.write("###### Ask a natural language question, and the system will retrieve the most relevant image/document snippet and generate an answer using **Gemini**.")

if st.session_state.image_paths:
    question = st.text_input("❓ Type your question here:")
    if st.button("🔍 Run Vision RAG") and question:
        top_image = search(question, co, st.session_state.doc_embeddings, st.session_state.image_paths)
        if top_image:
            # Display image and answer in two columns
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("<div class='image-card'>", unsafe_allow_html=True)
                st.image(top_image, caption=f"📌 Relevant result", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with col2:
                final_answer = answer(question, top_image, genai_client)
                st.markdown(f"<div class='answer-box'>🤖 <b>Gemini Answer:</b><br>{final_answer}</div>", unsafe_allow_html=True)
else:
    st.warning("⚠️ Upload files first to enable Q&A.")

# --- Footer ---
st.markdown("<div class='footer'>✨ Built with Cohere Embed-4 & Google Gemini | A Next-Gen Vision RAG</div>", unsafe_allow_html=True)
