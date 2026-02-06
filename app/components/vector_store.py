from langchain_community.vectorstores import FAISS 
import os
from app.components.embeddings import get_embedding_model

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

from app.config.config import DB_FAISS_PATH

logger = get_logger(__name__)

def load_vector_store():
    try:
        embedding_model = get_embedding_model()
        
        if os.path.exists(DB_FAISS_PATH):
            logger.info("Loading existing FAISS vector store...")
            return FAISS.load_local(
                DB_FAISS_PATH, embedding_model,
                allow_dangerous_deserialization=True 
            )
        else:
            logger.warning("No vector store found at the specified path. Returning None.")

    except Exception as e:
        error_message = f"Error loading vector store: {str(e)}"
        logger.error(error_message)

# Creatibng new vector store 
def save_vector_store(text_chunks):
    try:
        if not text_chunks:
            raise CustomException("No text chunks provided to create the vector store.")
        
        logger.info("Creating new FAISS vector store...")
        embedding_model = get_embedding_model()
        db = FAISS.from_documents(text_chunks, embedding_model)

        logger.info("Saving FAISS vector store to disk...")
        db.save_local(DB_FAISS_PATH)

        logger.info("FAISS vector store saved successfully.")
        return db 
    except Exception as e:
        error_message = f"Failed to create or save vector store: {str(e)}"
        logger.error(error_message)