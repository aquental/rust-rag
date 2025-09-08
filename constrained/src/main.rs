mod data;
mod embeddings;
mod vector_db;
mod llm;

use std::env;
use std::error::Error;
use data::{load_and_chunk_dataset, Chunk};
use embeddings::SentenceEmbedder;
use vector_db::build_chroma_collection;
use llm::LlmClient;
use chromadb::collection::QueryOptions;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // 1. Load and chunk documents with a chunk size of 50
    let current_dir = env::current_dir()?;
    let dataset_file = current_dir.join("data").join("corpus.json");
    let docs: Vec<Chunk> = load_and_chunk_dataset(dataset_file.to_str().unwrap(), 50)?;

    // 2. Initialize embedder and build collection
    let embedder = SentenceEmbedder::new().await?;
    let collection = build_chroma_collection(&docs, "corpus_collection", &embedder).await?;
    println!("ChromaDB collection created with {} document chunks.", collection.count().await?);

    // 3. Prepare LLM client
    let llm = LlmClient::new();

    // 4. Run sample query
    let query = "Highlight the main policies that apply to employees.";
    let query_embeddings = embedder.embed_texts(&[query])?;

    let query_opts = QueryOptions {
        query_texts: None,
        query_embeddings: Some(query_embeddings),
        n_results: Some(2),
        where_metadata: None,
        where_document: None,
        include: Some(vec!["documents".into(), "distances".into()]),
    };

    let retrieval = collection.query(query_opts, None).await?;

    // 5. Safely extract documents from the first group
    let fallback_docs = Vec::new();
    let docs_ref = retrieval
        .documents
        .as_ref()
        .and_then(|groups| groups.get(0))
        .unwrap_or(&fallback_docs);

    let retrieved_context = if docs_ref.is_empty() {
        "".to_string()
    } else {
        docs_ref.iter()
            .map(|txt| format!("- {}", txt))
            .collect::<Vec<_>>()
            .join("\n")
    };

    // 6. Run constrained generation with all strategies
    for strategy in &["base", "strict", "cite"] {
        println!("=== Strategy: {} ===", strategy);
        let (answer, used_context) = llm
            .generate_with_constraints(query, &retrieved_context, strategy)
            .await?;
        println!("Constrained generation answer:\n{}\n", answer);
        println!("Context or lines used:\n{}\n", used_context);
    }

    Ok(())
}
