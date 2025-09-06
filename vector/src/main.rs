mod data;
mod embeddings;
mod vector_db;

use crate::embeddings::SentenceEmbedder;
use chromadb::collection::CollectionEntries;
use data::{Chunk, load_and_chunk_dataset};
use serde_json::{Value, json};
use std::env;
use std::error::Error;
use vector_db::build_chroma_collection;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let current_dir = env::current_dir()?;
    let dataset_file = current_dir.join("data").join("corpus.json");
    let embedder = SentenceEmbedder::new().await?;

    println!("Loading data from: {}", dataset_file.display());

    // Build the initial collection from chunked documents
    let chunked_docs = load_and_chunk_dataset(dataset_file.to_str().unwrap(), 30)?;
    let collection = build_chroma_collection(&chunked_docs, "corpus_collection", &embedder).await?;
    let total_docs = collection.count().await?;
    println!("ChromaDB collection created with {} documents.", total_docs);

    // Create a new document
    let new_chunk = Chunk {
        doc_id: 99,
        chunk_id: 0,
        category: "food".to_string(),
        text: "Bananas are yellow fruits rich in potassium.".to_string(),
    };

    // Generate a unique ID string
    let doc_id = format!("chunk_{}_{}", new_chunk.doc_id, new_chunk.chunk_id);

    // Add the new document to the collection
    let texts_for_embedding = vec![new_chunk.text.clone()];
    let texts: Vec<&str> = texts_for_embedding.iter().map(|s| s.as_str()).collect();
    let ids = vec![doc_id.as_str()];
    let metadatas: Vec<serde_json::Map<String, Value>> = vec![
        json!({
            "doc_id": new_chunk.doc_id,
            "chunk_id": new_chunk.chunk_id,
            "category": new_chunk.category
        })
        .as_object()
        .unwrap()
        .clone(),
    ];
    let embeddings = embedder.embed(&texts).await?;

    let entries = CollectionEntries {
        ids,
        embeddings: Some(embeddings),
        metadatas: Some(metadatas),
        documents: Some(texts),
    };

    collection.upsert(entries, None).await?;

    // Print the updated document count
    let updated_count = collection.count().await?;
    println!("Document count after adding new chunk: {}", updated_count);

    // Remove the newly added document
    collection
        .delete(Some(vec![doc_id.as_str()]), None, None)
        .await?;

    // Print the final document count
    let final_count = collection.count().await?;
    println!("Document count after deletion: {}", final_count);

    Ok(())
}
