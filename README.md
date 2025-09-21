#  AI Vision RAG Explorer

### An interactive **Visual Retrieval-Augmented Generation (RAG)** system that combines:  
### 🔎 **Cohere Embed-4** for multimodal embeddings  
### ⚡ **Google Gemini 2.5 Flash** for visual question answering  

### Built with **Streamlit**, this app allows you to upload **images** and **PDFs**, then query them with natural language to extract insights from charts, diagrams, and document pages.  

## ✨ Features
### 📂 **Upload PDFs & Images**  
  * PDFs are automatically converted into **page images**.  
  * Images are auto-enhanced for better embedding + Q&A.  

### 🔎 **Multimodal Retrieval**  
  * Uses **Cohere Embed-4** to compute embeddings for each image/page.  
  * Finds the most semantically relevant page/image for a given query.  

### 🤖 **AI-Powered Answers**  
  * Google **Gemini 2.5 Flash** analyzes the retrieved visual content.  
  * Generates clear, context-aware answers to your question.  

### 🎨 **Modern UI/UX**  
  * Gradient background, styled buttons, answer bubbles, image cards.  
  * Dual-column layout → relevant image on the left, Gemini answer on the right.  

### 🧠 **Session Management**  
  * Stores uploaded files and embeddings in **Streamlit session state**.  
  * Enables multiple queries without re-uploading documents.  


## ⚙️ Requirements
* Python **3.9+**
* [Cohere API key](https://dashboard.cohere.com/api-keys)  
* [Google Gemini API key](https://aistudio.google.com/app/apikey)  


## 🚀 Installation & Setup

### 1. Create Virtual Environment
```bash
python -m venv venv
.\venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install streamlit cohere google-genai python-dotenv PyMuPDF pillow numpy
```

### 3. Add API Keys
### Create a .env file in the root folder:
```bash
COHERE_API_KEY=your_cohere_api_key
GEMINI_API_KEY=your_gemini_api_key
```

### 4. Run Streamlit App
```bash
streamlit run app.py
```

## 🔍 How It Works  

### 1. 📥 **Upload PDFs or Images**  
   * PDFs → Each page is rendered to an image using **PyMuPDF**.  
   * Images → Automatically enhanced and resized for better processing.  

### 2. 🧮 **Embedding with Cohere Embed-4**  
   * Generates dense **multimodal embeddings** for every image or PDF page.  

### 3. ❓ **Ask a Natural Language Question**  
   * Your query is embedded using **search_query mode**.  
   * **Cosine similarity** retrieves the most relevant image/page.  

### 4. ⚡ **Answer Generation with Gemini**  
   * The retrieved image + your question are analyzed using **Google Gemini**.  
   * Produces a **context-aware AI-generated answer**.  

### 5. 🖼️ **Results**  
   * **Left panel** → Relevant image/page  
   * **Right panel** → Gemini’s generated answer  


## 📌 Example Use Cases  

* 📊 Extract insights from **financial charts**  
* 📑 Understand **tables and diagrams** in PDFs  
* 🔎 Ask targeted questions about **multi-page documents**  
* 🗂️ Perform **visual knowledge retrieval** from mixed content (images + PDFs)  



















