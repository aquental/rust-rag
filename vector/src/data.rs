use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs::File;
use std::io::BufReader;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub doc_id: usize,
    pub chunk_id: usize,
    pub category: String,
    pub text: String,
}

#[derive(Debug, Deserialize)]
struct Document {
    id: usize,
    content: String,
    category: Option<String>,
}

/// Splits the given text into chunks of size 'chunk_size' words.
pub fn chunk_text(text: &str, chunk_size: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut chunks = Vec::new();

    for i in (0..words.len()).step_by(chunk_size) {
        let end = (i + chunk_size).min(words.len());
        let chunk = words[i..end].join(" ");
        chunks.push(chunk);
    }

    chunks
}

/// Loads a dataset from JSON file_path, then splits each document into smaller chunks.
pub fn load_and_chunk_dataset(
    file_path: &str,
    chunk_size: usize,
) -> Result<Vec<Chunk>, Box<dyn Error>> {
    // Open and load the JSON file
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let documents: Vec<Document> = serde_json::from_reader(reader)?;

    let mut all_chunks = Vec::new();

    for doc in documents.iter() {
        // Extract category and id from the document
        let doc_id = doc.id;
        let category = doc
            .category
            .clone()
            .unwrap_or_else(|| "unknown".to_string());

        // Split document content into chunks
        let doc_chunks = chunk_text(&doc.content, chunk_size);

        // Create Chunk struct for each chunk
        for (chunk_id, chunk_str) in doc_chunks.into_iter().enumerate() {
            let chunk = Chunk {
                doc_id,
                chunk_id,
                category: category.clone(),
                text: chunk_str,
            };
            all_chunks.push(chunk);
        }
    }

    Ok(all_chunks)
}
