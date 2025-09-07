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

    // Define a user query and category for filtering
    let user_query = "What are the recent developments in artificial intelligence?";
    let category_filter = None;  // Options: "Technology", "Science", "Health", etc., or None
    let distance_threshold = Some(1.0);  // Only include chunks with distance <= 1.0 (good similarity)
                                         // Typical ranges: 0.0-0.5 (very similar), 0.5-1.0 (similar), 1.0-1.5 (somewhat similar), >1.5 (dissimilar)

    // Retrieve the top documents relevant to the query with both filters
    let top_k = 3;

    println!("\n{}", "=".repeat(60));
    println!("RAG SYSTEM WITH DUAL FILTERING");
    println!("{}", "=".repeat(60));
    println!("Query: {}", user_query);
    println!("Category Filter: {:?}", category_filter.unwrap_or("None"));
    println!("Distance Threshold: {:?} (lower = more similar)", distance_threshold.unwrap_or(2.0));
    println!("Max Results: {}", top_k);
    println!("{}", "=".repeat(60));

    let retrieved_chunks = retrieve_top_chunks(
        &collection, 
        user_query, 
        top_k, 
        &embedder, 
        category_filter,
        distance_threshold
    ).await?;

    // Check if we found any results
    if retrieved_chunks.is_empty() {
        println!("\n⚠️  No relevant documents found!");
        println!("\nThe search returned no results that meet your criteria:");
        if let Some(category) = category_filter {
            println!("  • Category: {}", category);
        }
        if let Some(threshold) = distance_threshold {
            println!("  • Similarity threshold: distance ≤ {:.2}", threshold);
        }
        
        println!("\nSuggestions:");
        println!("  1. Try relaxing the distance threshold (increase the value)");
        println!("  2. Remove or change the category filter");
        println!("  3. Rephrase your query");
        
        // Try without filters to show what's available
        println!("\n{}", "-".repeat(60));
        println!("Attempting search without filters for comparison...");
        let unfiltered_chunks = retrieve_top_chunks(
            &collection,
            user_query,
            top_k,
            &embedder,
            None,  // No category filter
            None   // No distance threshold
        ).await?;
        
        if !unfiltered_chunks.is_empty() {
            println!("\nFound {} documents without filters:", unfiltered_chunks.len());
            for (i, chunk) in unfiltered_chunks.iter().enumerate() {
                println!("\n  {}. Distance: {:.4}, Doc ID: {}", i + 1, chunk.distance, chunk.doc_id);
                println!("     Preview: {}...", &chunk.chunk[..chunk.chunk.len().min(150)]);
            }
            println!("\nThese results show what's available without filtering constraints.");
        } else {
            println!("\nNo documents found even without filters. The query might be too specific.");
        }
    } else {
        println!("\n✓ Retrieved {} documents meeting all criteria:", retrieved_chunks.len());
        
        // Display retrieved chunks with details
        for (i, chunk) in retrieved_chunks.iter().enumerate() {
            println!("\n{}", "-".repeat(40));
            println!("Document {} | ID: {} | Distance: {:.4}", i + 1, chunk.doc_id, chunk.distance);
            println!("Similarity: {}", match chunk.distance {
                d if d <= 0.5 => "Very High ★★★★★",
                d if d <= 0.8 => "High ★★★★",
                d if d <= 1.0 => "Good ★★★",
                d if d <= 1.2 => "Moderate ★★",
                _ => "Low ★"
            });
            println!("{}", "-".repeat(40));
            println!("{}", chunk.chunk);
        }

        // Build the LLM prompt using the retrieved contexts
        println!("\n{}", "=".repeat(60));
        println!("GENERATING LLM RESPONSE");
        println!("{}", "=".repeat(60));
        
        let llm_client = LlmClient::new();
        let final_prompt = llm_client.build_prompt(user_query, &retrieved_chunks);
        
        // Show prompt preview (first part)
        println!("\nPrompt Preview:");
        let prompt_preview = if final_prompt.len() > 500 {
            format!("{}...", &final_prompt[..500])
        } else {
            final_prompt.clone()
        };
        println!("{}", prompt_preview);

        // Query the LLM
        match llm_client.get_llm_response(&final_prompt).await {
            Ok(answer) => {
                println!("\n{}", "=".repeat(60));
                println!("LLM RESPONSE");
                println!("{}", "=".repeat(60));
                println!("{}", answer);
                println!("{}", "=".repeat(60));
            }
            Err(e) => {
                eprintln!("\n❌ Error getting LLM response: {}", e);
                eprintln!("   Make sure OPENAI_API_KEY is set in your environment or .env file");
            }
        }
    }

    Ok(())
}
