mod data;
mod embeddings;
mod vector_db;
mod hybrid;

use data::load_and_chunk_dataset;
use embeddings::SentenceEmbedder;
use hybrid::{hybrid_retrieval, Bm25Index};
use std::env;
use std::error::Error;
use vector_db::build_chroma_collection;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // 1) Load & chunk
    let dataset_file = env::current_dir()?.join("data").join("corpus.json");
    println!("Loading data from: {}", dataset_file.display());
    let chunks = load_and_chunk_dataset(dataset_file.to_str().unwrap(), 40)?;

    // 2) Build BM25 index
    let bm25 = Bm25Index::new(&chunks);

    // 3) Build dense collection & embedder
    let embedder = SentenceEmbedder::new().await?;
    let collection = build_chroma_collection(&chunks, "hybrid_collection", &embedder).await?;
    println!("Hybrid collection has {} documents.", collection.count().await?);

    // 4) Perform hybrid retrieval
    let query = "What do our internal company policies state?";
    let results = hybrid_retrieval(
        query,
        &chunks,
        &bm25,
        &collection,
        /* top_k */ 3,
        /* alpha  */ 0.6,
        &embedder,
    )
        .await?;

    if results.is_empty() {
        println!("No chunks found. Fallback to apology.");
    } else {
        println!("Final hybrid top‑k results:");
        for (idx, score) in results {
            println!(" → [{}] (score {:.4}) {}", idx, score, chunks[idx].text);
        }
    }

    Ok(())
}
