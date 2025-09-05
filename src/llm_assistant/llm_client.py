"""Unified wrapper around multiple LLM providers."""
from __future__ import annotations

import os
from typing import List, Optional, Dict

from dotenv import load_dotenv

# Provider-specific imports
import openai
from huggingface_hub import InferenceClient
import google.generativeai as genai


class LLMClient:
    """Handle text generation for different LLM providers.

    Supported providers are OpenAI, Hugging Face Inference API, and
    Google's Generative AI models. The provider can be selected via the
    ``provider`` argument or the ``LLM_PROVIDER`` environment variable.
    """

    def __init__(self, provider: Optional[str] = None) -> None:
        load_dotenv()
        self.provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()

        if self.provider == "openai":
            openai.api_key = os.getenv("OPENAI_API_KEY", "")
            self.model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        elif self.provider == "huggingface":
            token = os.getenv("HF_TOKEN", "")
            self.model = os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
            self.client = InferenceClient(token=token)
        elif self.provider == "google":
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY", ""))
            self.model = os.getenv("GOOGLE_MODEL", "gemini-1.5-flash-latest")
            self.client = genai.GenerativeModel(self.model)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def generate(self, prompt: str) -> str:
        """Generate text from ``prompt`` using the configured provider."""
        if self.provider == "openai":
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response["choices"][0]["message"]["content"].strip()
        if self.provider == "huggingface":
            return self.client.text_generation(prompt, model=self.model).strip()
        if self.provider == "google":
            return self.client.generate_content(prompt).text.strip()

        raise RuntimeError("No provider configured")

    def chat_completion(self, messages: List[Dict[str, str]]) -> str:
        """Chat-based completion using the configured provider."""
        if self.provider == "openai":
            response = openai.ChatCompletion.create(model=self.model, messages=messages)
            return response["choices"][0]["message"]["content"].strip()

        # fall back to simple text generation for providers without native chat APIs
        prompt = "\n".join(m["content"] for m in messages)
        if self.provider == "huggingface":
            return self.client.text_generation(prompt, model=self.model).strip()
        if self.provider == "google":
            return self.client.generate_content(prompt).text.strip()

        raise RuntimeError("No provider configured")
