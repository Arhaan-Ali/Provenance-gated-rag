#Local Rag which uses Locally downloaded LLMs instead of open API keys with llamaIndex

import os
import sys

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

DATA_DIR = "data"
PERSIST_DIR = "storage"
RESULTS_DIR = "results"
OLLAMA_URL = "http://localhost:11434"

# Settings

Settings.llm = Ollama(
    # this implementation uses qwen3: 30b-a3b any other Local LLM will also be supported
    model="qwen3:30b-a3b",
    base_url=OLLAMA_URL,

    request_timeout=600.0,

    context_window=32768,

    thinking=False, #thinking is false so that qwen 30b's internal thinking is not shared as answer.

    temperature=0.1,
)

# qwen3: 30b-a3b is not sufficient or advised for embeddings so using a nomic-embed-txt embedder from huggin-face
Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url=OLLAMA_URL,
)


Settings.node_parser = SentenceSplitter(chunk_size=800, chunk_overlap=100)



def get_index() -> VectorStoreIndex:
    if os.path.isdir(PERSIST_DIR):
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        return load_index_from_storage(storage_context)

    if not os.path.isdir(DATA_DIR):
        sys.exit(f"Put your documents in ./{DATA_DIR}/ first.")

    documents = SimpleDirectoryReader(DATA_DIR, recursive=True).load_data()
    print(f"Loaded {len(documents)} documents. Embedding...")
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    return index

#can save answer from save with just writing the name of the file
def save_last(last: dict, name: str) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{name}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"MODEL:    {Settings.llm.model}\n")
        f.write(f"QUESTION: {last['q']}\n\n")
        f.write(f"{last['a']}\n\n")
        f.write("--- Sources ---\n")
        for score, fname, text in last["s"]:
            f.write(f"\n[{score:.3f}] {fname}\n{text[:500].strip()}\n")
    print(f"Saved to {path}\n")



def main() -> None:
    index = get_index()
    query_engine = index.as_query_engine(
        similarity_top_k=5,
        response_mode="compact",
        streaming=False,
    )

    last = None

    print("\nAsk a question. Type  save <name>  to keep the last answer.")
    print("Blank line or Ctrl-C to quit.\n")

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break

        if line.startswith("save "):
            name = line[5:].strip()
            if last is None:
                print("Nothing to save yet.\n")
            elif not name:
                print("Give it a name: save my_test_1\n")
            else:
                save_last(last, name)
            continue

        response = query_engine.query(line)
        print(f"\n{response}\n")

        srcs = []
        print("--- Sources ---")
        for node in response.source_nodes:
            fname = node.metadata.get("file_name", "?")
            score = float(node.score or 0)
            print(f"[{score:.3f}] {fname}: {node.text[:160].strip()}...")
            srcs.append((score, fname, node.text))
        print()

        last = {"q": line, "a": str(response), "s": srcs}


if __name__ == "__main__":
    main()