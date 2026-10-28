"""
LLM provider switch.

Every part of the project gets its LLM from this file, so changing from a
local Ollama model to a cloud model is just a change in `.env`:

    TEXT_LLM_PROVIDER=ollama              -> local model via Ollama (default)
    TEXT_LLM_PROVIDER=openai_compatible   -> any OpenAI-style cloud API

The same applies to the VISION_LLM_* settings for the image model.
"""

import base64
import json
import re
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from restaurant_ai import config


def _build_chat_model(provider: str, model: str, api_key: str, base_url: str,
                      temperature: float, json_mode: bool) -> BaseChatModel:
    """Create a LangChain chat model for the chosen provider."""
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            base_url=config.OLLAMA_BASE_URL,
            temperature=temperature,
            num_ctx=config.OLLAMA_NUM_CTX,  # context window; Ollama's small default can cut long prompts
            format="json" if json_mode else None,  # Ollama "JSON mode"
            reasoning=False,  # turn off "thinking" for models like qwen3 (faster)
        )

    if provider == "openai_compatible":
        from langchain_openai import ChatOpenAI

        if not api_key:
            raise ValueError("An API key is required for the 'openai_compatible' provider. "
                             "Set TEXT_LLM_API_KEY / VISION_LLM_API_KEY in your .env file.")
        extra = {"response_format": {"type": "json_object"}} if json_mode else {}
        return ChatOpenAI(model=model, api_key=api_key, base_url=base_url,
                          temperature=temperature, model_kwargs=extra)

    raise ValueError(f"Unknown LLM provider '{provider}'. Use 'ollama' or 'openai_compatible'.")


def get_text_llm(temperature: float = 0.2, json_mode: bool = False) -> BaseChatModel:
    """Return the text LLM used by the data pipeline and the agents."""
    return _build_chat_model(config.TEXT_LLM_PROVIDER, config.TEXT_LLM_MODEL,
                             config.TEXT_LLM_API_KEY, config.TEXT_LLM_BASE_URL,
                             temperature, json_mode)


def get_vision_llm(temperature: float = 0.2) -> BaseChatModel:
    """Return the vision (multimodal) LLM used for image captions."""
    return _build_chat_model(config.VISION_LLM_PROVIDER, config.VISION_LLM_MODEL,
                             config.VISION_LLM_API_KEY, config.VISION_LLM_BASE_URL,
                             temperature, json_mode=False)


# ---------------------------------------------------------------------------
# Small helper functions used across the project
# ---------------------------------------------------------------------------
def ask_llm(system_msg: str, prompt: str, json_mode: bool = False, temperature: float = 0.2) -> str:
    """Send a system message + user prompt to the text LLM and return the reply text."""
    llm = get_text_llm(temperature=temperature, json_mode=json_mode)
    response = llm.invoke([SystemMessage(content=system_msg), HumanMessage(content=prompt)])
    return str(response.content).strip()


async def ask_llm_async(system_msg: str, prompt: str, json_mode: bool = False, temperature: float = 0.2) -> str:
    """Async version of ask_llm (used by the agents so they can run in parallel)."""
    llm = get_text_llm(temperature=temperature, json_mode=json_mode)
    response = await llm.ainvoke([SystemMessage(content=system_msg), HumanMessage(content=prompt)])
    return str(response.content).strip()


def ask_vision_llm(system_msg: str, prompt: str, image_path: str | Path) -> str:
    """Send a prompt + an image to the vision LLM and return the caption text."""
    image_path = Path(image_path)
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"

    # Same message format as the course (OpenAI-style "image_url" block).
    # LangChain converts it to the right format for Ollama automatically.
    user_message = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
    ])
    response = get_vision_llm().invoke([SystemMessage(content=system_msg), user_message])
    return str(response.content).strip()


def parse_json(text: str):
    """
    Turn an LLM reply into a Python dict/list.

    Small local models sometimes wrap JSON in ```json fences or add a sentence
    before it, so we strip those before parsing.
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)  # remove markdown fences
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fall back to the first {...} or [...] block inside the text
        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise
