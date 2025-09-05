mod data;

use data::load_and_chunk_dataset;
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let current_dir = env::current_dir()?;
    let dataset_path = current_dir.join("data").join("corpus.json");

    // Call the chunking function with chunk_size = 10 and overlap = 2
    let chunks = load_and_chunk_dataset(dataset_path.to_str().ok_or("Invalid path")?, 10, 2)?;

    // Print how many chunks were created
    println!("Total chunks created: {}", chunks.len());

    // Print each chunk’s id and text
    for chunk in &chunks {
        println!(
            "Chunk ID: {}.{} - Text: {}",
            chunk.doc_id, chunk.chunk_id, chunk.text
        );
    }

    Ok(())
}
