import base64
import os
import sys
from venv import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from Backend.constants.prompts import rag_system_prompt, rag_user_prompt
from Backend.utils.chat_utils import detect_intent
from Backend.utils.llm_utils import call_openai, call_llama
from Backend.utils.rag_utils import retrieve_context


def chat_response(
    conversation_history: list[dict],
    latest_message: str,
    language: str
) -> dict:
    
    # 1. Intent detection
    is_rag_query, response_text, product_name, query = detect_intent(
        conversation_history, latest_message, language
    )

    # 2. Non-RAG conversational response
    if not is_rag_query:
        return {
            "status": "ok",
            "responses": [
                {"type": "text", "payload": response_text, "source": None}
            ],
        }

    print(f"Query : {query}")
    print(f"Product Name : {product_name}")
    # 3. Retrieve context and top-3 chunks for RAG
    context, top_k = retrieve_context(product_name, latest_message)

    # 4. RAG openAI call
    system_msg = {"role": "system", "content": rag_system_prompt}
    user_msg = {
        "role": "user",
        "content": rag_user_prompt.format(
            question=query, context=context, conversation_history=conversation_history, language=language
        ),
    }
    answer = call_openai([system_msg, user_msg])

    # 5. Assemble payload with response_type
    payload = {
        "status": "ok",
        "responses": [
            {"type": "text", "payload": answer, "source": None}
        ],
    }
    for _, doc, _ in top_k:
        if doc.get("type") == "image":
            try:
                with open(doc["image_filename"], "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                payload["responses"].append(
                    {
                        "type": "image",
                        "mime": "image/png",
                        "payload": b64,
                        "source": doc.get("source"),
                    }
                )
            except FileNotFoundError:
                pass

    return payload
