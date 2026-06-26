# RAG Medical Chatbot

An end-to-end retrieval augmented generation (RAG) chatbot for medical PDFs. The app loads PDF documents from `data/`, chunks and embeds them with Hugging Face embeddings, stores them in a FAISS vector store, and uses a Hugging Face hosted chat model to answer questions in a short, context-grounded format.

## Features

- PDF ingestion from a local `data/` directory
- Text chunking and FAISS vector storage
- Retrieval-augmented question answering
- Flask web interface with chat history
- Session-based conversation state and a chat clear endpoint

## Repository Layout

- `app/applications.py` - Flask entry point
- `app/components/` - loaders, embeddings, vector store, retriever, and LLM wiring
- `app/config/config.py` - model, path, and chunking settings
- `data/` - place source PDF files here
- `vectorstore/db_faiss/` - generated FAISS index
- `templates/index.html` - UI template

## Prerequisites

- Python 3.10 or newer
- A Hugging Face access token with inference access
- PDF documents to index in `data/`

## Setup

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -e .
```

Create a `.env` file in the project root and add your Hugging Face token:

```env
HF_TOKEN=your_hugging_face_token_here
```

## Prepare the Knowledge Base

Copy one or more PDF files into the `data/` directory, then build the vector store:

```bash
python -m app.components.data_loader
```

This creates or updates the FAISS index in `vectorstore/db_faiss/`.

## Run the App

Start the Flask application from the project root:

```bash
python app/applications.py
```

Then open your browser at:

```text
http://localhost:5000
```

## How It Works

1. The PDF loader reads documents from `data/`.
2. The text splitter breaks them into overlapping chunks.
3. The embedding model converts chunks into vectors.
4. FAISS stores the vectors for fast retrieval.
5. The Flask app retrieves the most relevant chunk and asks the LLM to answer in 2-3 lines.

## Configuration

The main runtime settings live in `app/config/config.py`:

- `HF_TOKEN` - Hugging Face token from the environment
- `HUGGINGFACE_REPO_ID` - model repository used by the chat chain
- `DATA_PATH` - directory that holds input PDFs
- `DB_FAISS_PATH` - output path for the vector store
- `CHUNK_SIZE` and `CHUNK_OVERLAP` - document chunking settings

## Notes

- The app keeps chat messages in the user session.
- Use the `/clear` route to reset the conversation.
- If the vector store is missing, create it before launching the Flask app.

## Troubleshooting

- If the app cannot find documents, confirm that the `data/` folder exists and contains PDF files.
- If model loading fails, verify that `HF_TOKEN` is set and valid.
- If retrieval fails, rebuild the vector store after changing the source PDFs.
