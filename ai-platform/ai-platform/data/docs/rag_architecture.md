# RAG Architecture

Documents are split into chunks of roughly 600 characters on paragraph boundaries. Each chunk keeps the name of its source file for citation.

Embeddings are produced with a TF-IDF vectorizer using unigrams and bigrams, so the project runs offline without any API key. The Embedder interface can be replaced by a neural embedding model or an embeddings API.

The vector store keeps a normalized matrix in memory and ranks chunks by cosine similarity. For production scale, replace it with pgvector inside PostgreSQL or a dedicated vector database.

The generator receives the top retrieved chunks and must answer only from them. If the best similarity score is too low the pipeline answers that it does not know.
