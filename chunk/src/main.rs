mod data;

use data::load_and_chunk_dataset;
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let current_dir = env::current_dir()?;
    let dataset_path = current_dir.join("data").join("corpus.json");

    // TODO: Define keywords to track
    let keywords: &[&str] = &["testing", "chunking"];
    // TODO: Call load_and_chunk_dataset with the keywords
    let chunked_docs = load_and_chunk_dataset(dataset_path.to_str().unwrap(), 30, keywords)?;
    // TODO: Print out each chunk's text and found keywords
    for chunk in chunked_docs {
        println!(
            "doc_id: {}, chunk_id: {}, category: {}\n{}",
            chunk.doc_id, chunk.chunk_id, chunk.category, chunk.text
        );
    }

    Ok(())
}
