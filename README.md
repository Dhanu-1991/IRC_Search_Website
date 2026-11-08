# IRC Search Assistant

A Streamlit website for asking questions about Indian Roads Congress (IRC)
standards. The app retrieves relevant chunks from indexed PDF documents and
uses Groq to generate an answer grounded in those sources.

## Setup

1. Install Python 3.11 or newer.
2. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env` and add your `GROQ_API_KEY`.
5. Start the website:

   ```powershell
   streamlit run app.py
   ```

Open the local URL shown by Streamlit. Use **Add IRC PDFs** in the sidebar to
upload documents and build the searchable index. The generated vector store is
kept in `data/vector_store/`, which is intentionally ignored by Git.
