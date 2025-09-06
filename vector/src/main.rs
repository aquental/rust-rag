mod data;
mod embeddings;
mod vector_db;

use data::load_and_chunk_dataset;
use vector_db::build_chroma_collection;
use std::error::Error;
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {

    let current_dir = env::current_dir()?;
    let dataset_file = current_dir.join("data").join("corpus.json");

    println!("Loading data from: {}", dataset_file.display());

    let chunked_docs = load_and_chunk_dataset(dataset_file.to_str().unwrap(), 50)?;
    println!("Loaded {} document chunks", chunked_docs.len());

    let collection = build_chroma_collection(&chunked_docs, "corpus_collection").await?;

    let total_docs = collection.count().await?;

    println!("ChromaDB collection created with {} documents.", total_docs);

    Ok(())
}
