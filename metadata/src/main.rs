mod data;
mod embeddings;
mod vector_db;

use data::load_and_chunk_dataset;
use embeddings::SentenceEmbedder;
use std::env;
use std::error::Error;
use vector_db::{build_chroma_collection, metadata_enhanced_search};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Initialize the sentence embedder
    let embedder = SentenceEmbedder::new().await?;

    // Load sample data from JSON file
    let current_dir = env::current_dir()?;
    let dataset_file = current_dir.join("data").join("corpus.json");
    println!("Loading data from: {}", dataset_file.display());

    // Load and chunk the documents
    let chunked_docs = load_and_chunk_dataset(dataset_file.to_str().unwrap(), 30)?;

    // Create or get collection and add documents
    let collection =
        build_chroma_collection(&chunked_docs, "metadata_demo_collection", &embedder).await?;
    println!(
        "ChromaDB collection created with {} documents.",
        collection.count().await?
    );

    // Define query
    let query_input = "Recent advancements in AI and their impact on teaching";

    // Search WITHOUT category filtering
    println!("\n======== WITHOUT CATEGORY FILTER ========");
    let no_filter_results =
        metadata_enhanced_search(&collection, query_input, None, 3, &embedder).await?;

    for chunk in no_filter_results {
        println!(
            "Doc ID: {}, Category: {}, Distance: {:.4}",
            chunk.doc_id,
            chunk.category.unwrap_or_else(|| "Unknown".to_string()),
            chunk.distance
        );
        println!("Chunk: {}\n", chunk.chunk);
    }

    // Search WITH category filtering
    println!("\n======== WITH CATEGORY FILTER (Education) ========");
    let filter_results = metadata_enhanced_search(
        &collection,
        query_input,
        Some(vec!["Education".to_string()]),
        3,
        &embedder,
    )
    .await?;

    for chunk in filter_results {
        println!(
            "Doc ID: {}, Category: {}, Distance: {:.4}",
            chunk.doc_id,
            chunk.category.unwrap_or_else(|| "Unknown".to_string()),
            chunk.distance
        );
        println!("Chunk: {}\n", chunk.chunk);
    }

    Ok(())
}
