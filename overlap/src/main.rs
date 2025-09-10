mod data;
mod embeddings;
mod vector_db;
mod summary;
mod llm;

use std::env;
use std::error::Error;
use data::load_and_chunk_dataset;
use embeddings::SentenceEmbedder;
use vector_db::build_chroma_collection;
use llm::LlmClient;
use summary::{are_chunks_overlapping, summarize_chunks};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // 1) Load & chunk
    let dataset_file = env::current_dir()?.join("data").join("corpus.json");
    println!("Loading data from: {}", dataset_file.display());
    let chunks = load_and_chunk_dataset(dataset_file.to_str().unwrap(), 40)?;

    // 2) Build collection
    let embedder = SentenceEmbedder::new().await?;
    let collection = build_chroma_collection(&chunks, "summary_demo_collection", &embedder).await?;
    println!("Collection has {} documents.", collection.count().await?);

    // 3) Query top 5
    let llm = LlmClient::new();
    let query = "Provide an overview of our internal policies.";
    let query_embeddings = embedder.embed_texts(&[query])?;
    let opts = chromadb::collection::QueryOptions {
        query_texts: None,
        query_embeddings: Some(query_embeddings),
        n_results: Some(5),
        where_metadata: None,
        where_document: None,
        include: Some(vec!["documents".into()]),
    };
    let result = collection.query(opts, None).await?;
    let docs = result.documents
        .and_then(|g| g.into_iter().next())
        .unwrap_or_default();

    if docs.is_empty() {
        println!("No chunks were retrieved for the query.");
        println!("Final answer:\nNo relevant information found.");
        return Ok(());
    }

    // 4) Decide summary vs list
    let texts: Vec<String> = docs.into_iter().collect();
    let context = if texts.len() > 3 || are_chunks_overlapping(&texts, 0.8) {
        summarize_chunks(&llm, &texts).await?
    } else {
        texts.into_iter().map(|t| format!("- {}", t)).collect::<Vec<_>>().join("\n")
    };

    // 5) Final answer
    let final_answer = llm.generate_final_answer(query, &context).await?;
    println!("Final answer:\n{}", final_answer);

    Ok(())
}
