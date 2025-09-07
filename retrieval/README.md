# Retrieval using embeddings

We'll delve into retrieving the most relevant chunks for any given query and building an LLM prompt to produce more accurate, context-driven answers.

## Retrieving the Most Relevant Chunks

Before your LLM can generate a coherent, context-rich answer, we need to fetch the right information. The vector database (Chroma) will rank which document chunks are most relevant for a given query.

# setup

```shell
chroma run --path ./chroma &
```
