from langchain_classic.chains import RetrievalQA

from langchain_core.prompts import PromptTemplate

from app.components.llm import load_llm
from app.components.vector_store import load_vector_store

from app.config.config import HUGGINGFACE_REPO_ID,HF_TOKEN
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from dotenv import load_dotenv

import os
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
logger = get_logger(__name__)

CUSTOM_PROMPT_TEMPLATE = """ Answer the following medical question in 2-3 lines maximum using only the information provided in the context.

Context:
{context}

Question:
{question}

Answer:
"""

def set_custom_prompt():
    return PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE,input_variables=["context" , "question"])

def create_qa_chain():
    try:
        logger.info("Loading vector store for context")
        db = load_vector_store()

        if db is None:
            raise CustomException("Vector store not present or empty")
        logger.info(f"Using HF model repo id: {HUGGINGFACE_REPO_ID}")
        logger.info(f"HF token present: {bool(HF_TOKEN)}")

        llm = load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID , hf_token=HF_TOKEN )

        if llm is None:
            raise CustomException("LLM not loaded")
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever = db.as_retriever(search_kwargs={'k':1}),
            return_source_documents=False,
            chain_type_kwargs={'prompt': set_custom_prompt()}
        )

        logger.info("Sucesfully created the QA chain")
        return qa_chain
    
    except Exception:
        logger.exception("Failed to make a QA chain")  # prints full traceback
        raise




