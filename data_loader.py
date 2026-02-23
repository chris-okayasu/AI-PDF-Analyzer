
import ollama
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter

EMBED_MODEL = "nomic-embed-text"

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=100)


def load_and_chunk_pdf(path: str):
    """
    Load a PDF file and split its content into overlapping text chunks.

    Args:
        path (str): Path to the PDF file.

    Returns:
        list[str]: List of text chunks.
    """
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings using a local Ollama model.

    Unlike OpenAI's API:
    - No external request leaves your machine.
    - Ollama must be running locally.
    - The embedding model must be downloaded beforehand.

    Args:
        texts (list[str]): List of text chunks.

    Returns:
        list[list[float]]: Embedding vectors.
    """
    embeddings = []

    for text in texts:
        response = ollama.embeddings(
            model=EMBED_MODEL,
            prompt=text
        )
        embeddings.append(response["embedding"])

    return embeddings