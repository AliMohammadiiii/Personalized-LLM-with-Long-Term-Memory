"""Interactive console for the personalized assistant."""

from sentence_transformers import SentenceTransformer

from llm_assistant import DialogueManager, LLMClient, UserMemoryModule


def main() -> None:
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    memory = UserMemoryModule(embedding_model)
    llm_client = LLMClient()
    dialogue = DialogueManager(memory, llm_client)

    while True:
        query = input("How can I help you? ")
        if query.lower() == "exit":
            break
        response = dialogue.get_response_and_learn(query)
        print(response)


if __name__ == "__main__":
    main()
