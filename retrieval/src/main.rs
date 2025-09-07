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

    // Define a query and category to test the retrieval function
    let user_query = "What are the recent discoveries and innovations?";
    let category_filter = None;  // Available categories: "Technology", "Science", "Health", "Education", "Business", "Transportation", etc., or None for no filter
    
    println!("\n{}", "=".repeat(60));
    println!("Query: {}", user_query);
    if let Some(category) = category_filter {
        println!("Category Filter: {}", category);
    } else {
        println!("Category Filter: None (searching all categories)");
    }
    println!("{}", "=".repeat(60));

    // Retrieve chunks matching the query and category
    let top_k = 3;
    let retrieved_chunks = retrieve_top_chunks(
        &collection, 
        user_query, 
        top_k, 
        &embedder,
        category_filter
    ).await?;

    // Handle cases where no chunks match the specified category
    if retrieved_chunks.is_empty() {
        println!("\n⚠️  No documents found matching the query");
        if let Some(category) = category_filter {
            println!("    with category filter: '{}'\n", category);
            println!("Trying without category filter...\n");
            
            // Try again without category filter
            let all_chunks = retrieve_top_chunks(
                &collection,
                user_query,
                top_k,
                &embedder,
                None
            ).await?;
            
            if !all_chunks.is_empty() {
                println!("Found {} documents without category filter:", all_chunks.len());
                for (i, chunk) in all_chunks.iter().enumerate() {
                    println!("\nDocument {}", i + 1);
                    println!("Text: {}...", &chunk.chunk[..chunk.chunk.len().min(200)]);
                    println!("Distance: {:.4}", chunk.distance);
                }
            } else {
                println!("No documents found even without category filter.");
            }
        } else {
            println!("No documents found in the collection.");
        }
        return Ok(());
    }

    // Print retrieved documents
    println!("\n✓ Retrieved {} documents:", retrieved_chunks.len());
    for (i, chunk) in retrieved_chunks.iter().enumerate() {
        println!("\n{}", "-".repeat(40));
        println!("Document {} (ID: {}, Distance: {:.4})", i + 1, chunk.doc_id, chunk.distance);
        println!("{}", "-".repeat(40));
        println!("{}", chunk.chunk);
    }

    // Create LLM client and generate response
    println!("\n{}", "=".repeat(60));
    println!("INITIALIZING LLM CLIENT");
    println!("{}", "=".repeat(60));
    let llm_client = LlmClient::new();
    
    // Build the prompt with retrieved chunks
    let prompt = llm_client.build_prompt(user_query, &retrieved_chunks);
    
    // Print the formatted prompt for demonstration
    println!("\n{}", "=".repeat(60));
    println!("FORMATTED PROMPT");
    println!("{}", "=".repeat(60));
    println!("{}", prompt);
    
    // Get LLM response
    println!("{}", "=".repeat(60));
    println!("GETTING LLM RESPONSE");
    println!("{}", "=".repeat(60));
    
    match llm_client.get_llm_response(&prompt).await {
        Ok(response) => {
            println!("\n{}", "=".repeat(60));
            println!("FINAL ANSWER");
            println!("{}", "=".repeat(60));
            println!("{}", response);
            println!("\n{}", "=".repeat(60));
        }
        Err(e) => {
            eprintln!("\n❌ Error getting LLM response: {}", e);
            eprintln!("   Make sure OPENAI_API_KEY is set in your environment or .env file");
        }
    }

    Ok(())
}
