mod data;

use data::load_and_chunk_dataset;
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let current_dir = env::current_dir()?;
    let dataset_path = current_dir.join("data").join("corpus.json");

    let chunked_docs = load_and_chunk_dataset(dataset_path.to_str().unwrap(), 30)?;
    println!(
        "Loaded and chunked {} chunks from dataset.",
        chunked_docs.len()
    );

    for chunk in chunked_docs {
        println!(
            "doc_id: {}, chunk_id: {}, category: {}\n{}",
            chunk.doc_id, chunk.chunk_id, chunk.category, chunk.text
        );
    }

    Ok(())
}
