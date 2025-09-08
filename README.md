# Rust RAG: A Simple Workflow

## Introduction

This guide presents a simple end-to-end RAG workflow in Rust, demonstrating how indexing, retrieval, prompt augmentation, and text generation combine to produce accurate answers.

## Project Chimera

Project Chimera serves as a fictional example to contrast a naive, context-free system that may generate inaccurate details with a RAG-based system that leverages an authoritative knowledge base for reliable responses.
We use simplified keyword matching to illustrate each component of the RAG pipeline, with plans to explore advanced techniques like embeddings and vector databases in future sections.

---

## encrypted .env

encrypt

```shell
gpg -c .env
```

decrypt

```shell
gpg -d .env.gpg > .env
```
