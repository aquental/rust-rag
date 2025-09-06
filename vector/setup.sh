#!/bin/sh

echo "Setting up environment..."

cd ~/projects/rust/rust-rag/vector
# Run the ChromaDB server in the background
chroma run --path ./chroma &
