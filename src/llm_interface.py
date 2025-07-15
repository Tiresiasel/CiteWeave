# llm_interface.py

import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from typing import Optional, List
import logging
import re
from src.config_manager import llm_config_selector

logger = logging.getLogger(__name__)

PROMPT_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")

def load_prompt_template(name: str) -> str:
    """
    Load a .txt prompt template from the prompts/ directory.
    The prompt must include {input} or other placeholders for formatting.
    """
    path = os.path.join(PROMPT_DIR, f"{name}.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prompt template not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

@llm_config_selector
def build_llm_chain(prompt_name: str, model_config=None, input_variables: Optional[list] = None) -> LLMChain:
    """
    Build a reusable LLMChain from a named prompt file.
    Automatically configures the model from config_manager.

    Args:
        prompt_name: Filename (without .txt) from prompts directory.
        model_config: Automatically injected by @llm_config_selector
        input_vars: List of variable names used in the prompt (e.g., ["input"])
    """
    prompt_str = load_prompt_template(prompt_name)
    input_variables = input_variables or ["input"]

    if model_config["provider"] == "openai":
        logger.info(f"Using OpenAI model: {model_config['model']}")
        llm = ChatOpenAI(model=model_config["model"])
    elif model_config["provider"] == "ollama":
        logger.info(f"Using Ollama model: {model_config['model']}")
        from langchain_ollama import Ollama
        llm = Ollama(model=model_config["model"])
    else:
        raise ValueError(f"Unsupported LLM provider: {model_config['provider']}")

    prompt = PromptTemplate(template=prompt_str, input_variables=input_variables)
    return LLMChain(llm=llm, prompt=prompt)

def extract_reference_section_via_llm(full_doc_text: str) -> List[str]:
    chunk = extract_reference_tail(full_doc_text, max_tokens=3500)
    chain = build_llm_chain("extract_reference_section", input_variables=["input"])
    try:
        raw_output = chain.run({"input": chunk}).strip()
        parsed = eval(raw_output) if raw_output.startswith("[") else []
        if isinstance(parsed, list) and all(isinstance(i, str) for i in parsed):
            return parsed
    except Exception as e:
        logger.exception("LLM extraction failed", exc_info=e)
    return []
    
def extract_reference_tail(full_doc_text: str, max_tokens: int = 14000) -> str:
    from tiktoken import get_encoding
    enc = get_encoding("cl100k_base")

    paras = full_doc_text.split("\n\n")
    start_idx = -1
    for i, p in enumerate(paras):
        if re.search(r'\b(references|bibliography)\b', p, re.IGNORECASE):
            start_idx = i
            break

    if start_idx == -1:
        logger.warning("No reference keyword found. Using last part of text as fallback.")
        text_to_search = "\n\n".join(paras[-10:])
    else:
        selected = []
        token_count = 0
        for p in paras[start_idx:]:
            selected.append(p)
            token_count = len(enc.encode("\n\n".join(selected)))
            if token_count > max_tokens:
                break
        text_to_search = "\n\n".join(selected)

    return text_to_search


if __name__ == "__main__":
    from PyPDF2 import PdfReader
    pdf_path = "test_files/Rivkin - 2000 - Imitation of Complex Strategies.pdf"
    pdf_reader = PdfReader(pdf_path)
    pdf_text = "\n".join([page.extract_text() for page in pdf_reader.pages])
    a = extract_reference_section_via_llm(pdf_text)
    print(a)