from langchain_community.embeddings import OllamaEmbeddings

def get_ollama_mxbai_embed_large_embedding_function():
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    return embeddings