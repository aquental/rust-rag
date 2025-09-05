mod data;

use data::load_and_chunk_dataset;
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let current_dir = env::current_dir()?;
    let dataset_path = current_dir.join("data").join("corpus.json");

    // Call the chunking function with the dataset file and a chunk size of 5
    let chunks = load_and_chunk_dataset(dataset_path.to_str().ok_or("Invalid path")?, 5)?;

    // Print how many chunks were created
    println!("Total chunks created: {}", chunks.len());

    // Print each chunk to inspect the results
    for chunk in chunks.iter() {
        println!(
            "Doc ID: {}, Chunk ID: {}, Category: {}, Text: '{}'",
            chunk.doc_id, chunk.chunk_id, chunk.category, chunk.text
        );
    }

    Ok(())
}
