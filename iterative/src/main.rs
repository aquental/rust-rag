
mod data;
mod embeddings;
mod vector_db;
mod retrieval;
mod llm;

use std::env;
use std::error::Error;


use data::{load_and_chunk_dataset, Chunk};
use embeddings::SentenceEmbedder;
use retrieval::{build_final_context, iterative_retrieval};
use vector_db::build_chroma_collection;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Load full documents
    let dataset_file = env::current_dir()?.join("data").join("corpus.json");
    println!("Loading data from: {}", dataset_file.display());
    let docs: Vec<Chunk> = load_and_chunk_dataset(dataset_file.to_str().unwrap(), 50)?;

    // Build collection
    let embedder = SentenceEmbedder::new().await?;
    let collection = build_chroma_collection(&docs, "iterative_collection", &embedder).await?;
    println!("ChromaDB collection created with {} documents.", collection.count().await?);

    // Iterative retrieval demo
    let initial_query = "What internal policies apply specifically to employees?";
    let iter_results = iterative_retrieval(
        &collection,
        &embedder,
        initial_query,
        /*steps=*/3,
        /*improvement_threshold=*/0.02,
    ).await?;

    // Build and print final context
    let final_context = build_final_context(&iter_results);
    println!("\nFinal combined context:\n{}", final_context);
    Ok(())
}
