import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


COLLECTION_NAME = "rag_course_lectures"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
QDRANT_URL = "http://localhost:6333"
TOP_K = 5
OPENAI_MODEL = "gpt-4o-mini"

# You can tune this threshold later after testing
MIN_RELEVANCE_SCORE = 0.45


def load_environment() -> None:
    project_root = Path(__file__).resolve().parent.parent
    env_path = project_root / ".env"

    load_dotenv(env_path)

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY was not found in .env file")


def get_embedding_model() -> SentenceTransformer:
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def get_qdrant_client() -> QdrantClient:
    print(f"Connecting to Qdrant at: {QDRANT_URL}")
    return QdrantClient(url=QDRANT_URL)


def retrieve_relevant_chunks(
    question: str,
    model: SentenceTransformer,
    client: QdrantClient,
    collection_name: str,
    top_k: int = TOP_K,
) -> list[dict]:
    print(f"Creating embedding for question: {question}")

    question_vector = model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()

    search_result = client.query_points(
        collection_name=collection_name,
        query=question_vector,
        limit=top_k,
    )

    results = []

    for hit in search_result.points:
        payload = hit.payload or {}
        results.append(
            {
                "score": hit.score,
                "source_type": payload.get("source_type", "unknown"),
                "file_name": payload.get("file_name", "unknown"),
                "chunk_index": payload.get("chunk_index", -1),
                "text": payload.get("text", ""),
            }
        )

    return results


def build_context(retrieved_chunks: list[dict]) -> str:
    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"""[Chunk {i}]
Source type: {chunk['source_type']}
File name: {chunk['file_name']}
Chunk index: {chunk['chunk_index']}
Similarity score: {chunk['score']}

Text:
{chunk['text']}
"""
        )

    return "\n\n".join(context_parts)


def is_context_good_enough(retrieved_chunks: list[dict], min_score: float = MIN_RELEVANCE_SCORE) -> bool:
    if not retrieved_chunks:
        return False

    best_score = retrieved_chunks[0]["score"]
    print(f"Best retrieval score: {best_score}")

    return best_score >= min_score


def generate_answer_from_rag(question: str, context: str) -> str:
    client = OpenAI()

    system_prompt = """
You are a helpful RAG assistant answering questions about lecture materials on GenAI, databases, and RAG.

Use ONLY the provided context to answer.
If the answer is not clearly present in the context, say that the answer was not found in the retrieved materials.
Be accurate, clear, and concise.
When possible, mention which source type or file the answer came from.
""".strip()

    user_prompt = f"""
Answer the following question using the retrieved context.

Question:
{question}

Retrieved context:
{context}
""".strip()

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response.choices[0].message.content or ""


def generate_answer_from_model_knowledge(question: str) -> str:
    client = OpenAI()

    system_prompt = """
You are a helpful assistant.

The local RAG database did not return sufficiently relevant lecture material.
Answer using your general knowledge.
You must clearly state that this answer is based on general model knowledge and not on the local database.
Do not pretend the answer came from the uploaded lecture files.
Be accurate, clear, and concise.
""".strip()

    user_prompt = f"""
The local knowledge base did not return strong enough results.

Please answer this question from general knowledge and begin with a short note such as:
"I could not find a confident answer in the local lecture database, so I am answering from general model knowledge."

Question:
{question}
""".strip()

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response.choices[0].message.content or ""


def print_retrieved_chunks(chunks: list[dict]) -> None:
    print("\n===== RETRIEVED CHUNKS =====")

    for i, chunk in enumerate(chunks, start=1):
        print(f"\nChunk {i}")
        print(f"Score: {chunk['score']}")
        print(f"Source type: {chunk['source_type']}")
        print(f"File name: {chunk['file_name']}")
        print(f"Chunk index: {chunk['chunk_index']}")
        print("Preview:")
        print(chunk["text"][:400])
        print("\n" + "=" * 60)


def main() -> None:
    load_environment()

    question = input("Enter your question: ").strip()
    if not question:
        raise ValueError("Question cannot be empty")

    embedding_model = get_embedding_model()
    qdrant_client = get_qdrant_client()

    retrieved_chunks = retrieve_relevant_chunks(
        question=question,
        model=embedding_model,
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        top_k=TOP_K,
    )

    print_retrieved_chunks(retrieved_chunks)

    if is_context_good_enough(retrieved_chunks):
        print("\nUsing local RAG context from Qdrant...")
        context = build_context(retrieved_chunks)
        answer = generate_answer_from_rag(question, context)
    else:
        print("\nLocal RAG context is not strong enough. Falling back to model knowledge...")
        answer = generate_answer_from_model_knowledge(question)

    print("\n===== FINAL ANSWER =====\n")
    print(answer)


if __name__ == "__main__":
    main()