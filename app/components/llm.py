# from langchain_huggingface import HuggingFaceEndpoint
# from app.config.config import HF_TOKEN,HUGGINGFACE_REPO_ID

# from app.common.logger import get_logger
# from app.common.custom_exception import CustomException

# logger = get_logger(__name__)

# def load_llm(huggingface_repo_id: str = HUGGINGFACE_REPO_ID , hf_token:str = HF_TOKEN):
#     try:
#         logger.info("Loading LLM from HuggingFace")

#         llm = HuggingFaceEndpoint(
#             repo_id=huggingface_repo_id,
#             huggingfacehub_api_token=hf_token,
#             temperature=0.3,
#             max_new_tokens=256,
#             return_full_text=False,
#             provider="auto",
#         )

#         logger.info("LLM loaded sucesfully...")

#         return llm
    
#     except Exception as e:
#         error_message = CustomException("Failed to load a llm" , e)
#         logger.error(str(error_message))



from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from app.config.config import HF_TOKEN, HUGGINGFACE_REPO_ID
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def load_llm(huggingface_repo_id: str = HUGGINGFACE_REPO_ID, hf_token: str = HF_TOKEN):
    try:
        logger.info("Loading LLM from HuggingFace (Inference Providers)")

        # IMPORTANT: use conversational for models that are only served as chat
        base = HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            huggingfacehub_api_token=hf_token,
            task="conversational",
            temperature=0.3,
            max_new_tokens=256,
            return_full_text=False,
            provider="novita",
        )

        llm = ChatHuggingFace(llm=base)

        logger.info("LLM loaded successfully.")
        return llm

    except Exception as e:
        error = CustomException("Failed to load an LLM", e)
        logger.error(str(error))
        raise  # don't swallow the exception
