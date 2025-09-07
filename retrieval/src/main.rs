mod data;
mod embeddings;
mod vector_db;
mod llm;

use data::load_documents;
use vector_db::{build_chroma_collection, retrieve_top_chunks};
use embeddings::SentenceEmbedder;
use llm::LlmClient;
use std::env;
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Determine the path to corpus.json
    let current_dir = env::current_dir()?;
    let dataset_file = current_dir.join("data").join("corpus.json");

    // Load documents without splitting them.
    let docs = load_documents(dataset_file.to_str().unwrap())?;

    // Create the embedder instance.
    let embedder = SentenceEmbedder::new().await?;

    // Build (or retrieve) the ChromaDB collection using full documents.
    let collection = build_chroma_collection(&docs, "full_document_collection", &embedder).await?;
    let doc_count = collection.count().await?;
    println!("ChromaDB collection created with {} documents.", doc_count);

    // TODO: Define a query string to test the retrieval function
    let user_query = "What are some recent technological breakthroughs?";

    // Retrieve the top documents relevant to the query.
    let top_k = 3;
    let retrieved_chunks = retrieve_top_chunks(&collection, user_query, top_k, &embedder).await?;

    // Print retrieved documents
    println!("\nRetrieved {} documents:", retrieved_chunks.len());
    for (i, chunk) in retrieved_chunks.iter().enumerate() {
        println!("Document {}", i + 1);
        println!("Text: {}", chunk.chunk);
        println!("Doc ID: {}", chunk.doc_id);
        println!("Distance: {}", chunk.distance);
        println!("{}", "-".repeat(40));
    }

    // Create LLM client and generate response
    println!("\n{}", "=".repeat(60));
    println!("Initializing LLM Client...");
    let llm_client = LlmClient::new();
    
    // Build the prompt with retrieved chunks
    let prompt = llm_client.build_prompt(user_query, &retrieved_chunks);
    
    // Print the formatted prompt for demonstration
    println!("\n{}", "=".repeat(60));
    println!("FORMATTED PROMPT:");
    println!("{}", "=".repeat(60));
    println!("{}", prompt);
    
    // Get LLM response
    println!("{}", "=".repeat(60));
    println!("Getting LLM Response...");
    println!("{}", "=".repeat(60));
    
    match llm_client.get_llm_response(&prompt).await {
        Ok(response) => {
            println!("\n{}", "=".repeat(60));
            println!("FINAL ANSWER:");
            println!("{}", "=".repeat(60));
            println!("{}", response);
            println!("{}", "=".repeat(60));
        }
        Err(e) => {
            eprintln!("Error getting LLM response: {}", e);
            eprintln!("Make sure OPENAI_API_KEY is set in your environment or .env file");
        }
    }

    Ok(())
}
