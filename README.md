# IRC Search Assistant

A Streamlit website for asking questions about one fixed Indian Roads Congress
document, `IRC_SP73.pdf`. The app retrieves relevant chunks from that document
and uses Groq to generate an answer grounded in those sources.

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

Open the local URL shown by Streamlit. On first startup, the app indexes
`data/fixed_document/IRC_SP73.pdf` automatically. The generated vector store
is kept in `data/vector_store/`, which is intentionally ignored by Git.

The fixed PDF must be present in the repository at:

```text
data/fixed_document/IRC_SP73.pdf
```

The PDF is indexed automatically when the deployed app starts and the vector
database is rebuilt locally on the server. Keep `.env` out of Git and add
`GROQ_API_KEY` through the hosting provider's secret settings.
