"""Dialogue management and retrieval-augmented generation."""

from __future__ import annotations

import logging
from typing import List, Optional

from .llm_client import LLMClient
from .memory import UserMemoryModule


class RAGCore:
    """Orchestrate retrieval and generation."""

    def __init__(self, umm: UserMemoryModule, llm_client: LLMClient) -> None:
        self.umm = umm
        self.llm_client = llm_client

    def generate_response(self, query: str, history: List[str]) -> str:
        memories = self.umm.retrieve_memory(query)
        memory_str = "\n- ".join(memories) if memories else ""
        history_str = "\n".join(history) if history else ""
        system_prompt = (
            "You are a personalized AI. Use these user Persona:\n- "
            f"{memory_str}\n\nAnd this history:\n{history_str}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        return self.llm_client.chat_completion(messages)


class DialogueManager:
    """Handle conversation flow and learn new facts."""

    def __init__(self, umm: UserMemoryModule, llm_client: LLMClient) -> None:
        self.umm = umm
        self.llm_client = llm_client
        self.rag_pipeline = RAGCore(umm, llm_client)
        self.history: List[str] = []

    def _extract_new_memory(self, user_input: str) -> Optional[str]:
        prompt = (
            "Analyze the user's statement. If it reveals a new personal fact "
            "(preference, detail, identity), state it concisely in the first "
            'person (e.g., "I live in Switzerland."). Otherwise, respond with '
            'ONLY the word "NO_FACT".\n'
            f'User statement: "{user_input}"\nNew fact:'
        )
        extracted = self.llm_client.chat_completion([{ "role": "user", "content": prompt }])
        if "NO_FACT" in extracted or not extracted:
            return None
        return extracted.strip('"')

    def get_response_and_learn(self, user_query: str) -> str:
        assistant_response = self.rag_pipeline.generate_response(user_query, self.history)
        self.history.extend([f"User: {user_query}", f"Assistant: {assistant_response}"])
        new_fact = self._extract_new_memory(user_query)
        if new_fact:
            logging.info(f"[Memory Update] New fact learned: '{new_fact}'")
            self.umm.add_memory([new_fact])
        return assistant_response
