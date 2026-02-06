import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

from app.config.config import DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP

logger = get_logger(__name__) # this will create a logger for this module

def load_pdf_files():
    try:
        if not os.path.exists(DATA_PATH):
            raise CustomException(f"Data path '{DATA_PATH}' does not exist.")
        
        logger.info(f"Loading PDF files from directory: {DATA_PATH}")

        loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)

        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents from PDF files.")

        if not documents:
            raise CustomException("No PDF documents found in the specified directory.")
        else:
            logger.info(f"Successfully loaded {len(documents)} PDF documents.")
    
        return documents
    
    except Exception as e:
        error_message = CustomException("Failed to load PDF files.", e)
        logger.error(str(error_message))
        raise []
    

def create_text_chunks(documents):
    try:
        if not documents:
            raise CustomException("No documents provided for text chunking.")
        logger.info(f"Creating text chunks from {len(documents)} documents with chunk size {CHUNK_SIZE} and overlap {CHUNK_OVERLAP}.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        text_chunks = text_splitter.split_documents(documents)

        logger.info(f"Created {len(text_chunks)} text chunks from the documents.")
        return text_chunks
    
    except Exception as e:
        error_message = CustomException("Failed to create text chunks from documents.", e)
        logger.error(str(error_message))
        raise []
    

    

    