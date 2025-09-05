use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub doc_id: usize,
    pub chunk_id: usize,
    pub category: String,
    pub text: String,
    pub keywords: HashSet<String>,
}

#[derive(Debug, Deserialize)]
struct Document {
    content: String,
    category: Option<String>,
}

/// Splits the given text into chunks of size 'chunk_size' words and tags matching keywords.
pub fn chunk_text(
    text: &str,
    chunk_size: usize,
    keywords: &[&str],
) -> Vec<(String, HashSet<String>)> {
    // Split the text into words
    let words: Vec<&str> = text.split_whitespace().collect();

    // Convert keywords to a HashSet for efficient lookup
    let keyword_set: HashSet<String> = keywords.iter().map(|&k| k.to_lowercase()).collect();
    let mut chunks = Vec::new();

    // Iterate over the text in steps of `chunk_size`
    for i in (0..words.len()).step_by(chunk_size) {
        let end = (i + chunk_size).min(words.len());
        // Join chunk into a string
        let chunk_text = words[i..end].join(" ");

        // Scan for matching keywords
        let mut matched_keywords = HashSet::new();
        for word in chunk_text.split_whitespace() {
            if keyword_set.contains(&word.to_lowercase()) {
                matched_keywords.insert(word.to_string());
            }
        }

        // Add chunk and its matched keywords to the result
        chunks.push((chunk_text, matched_keywords));
    }

    chunks
}

/// Loads a dataset from JSON file_path, then splits each document into smaller chunks.
pub fn load_and_chunk_dataset(
    file_path: &str,
    chunk_size: usize,
    keywords: &[&str],
) -> Result<Vec<Chunk>, Box<dyn Error>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let documents: Vec<Document> = serde_json::from_reader(reader)?;

    let mut all_chunks = Vec::new();

    for (doc_id, doc) in documents.iter().enumerate() {
        let doc_text = &doc.content;
        let doc_category = doc
            .category
            .clone()
            .unwrap_or_else(|| "general".to_string());

        // Call chunk_text and get chunk-string + keyword set pairs
        let doc_chunks = chunk_text(doc_text, chunk_size, keywords);

        // Iterate through and collect Chunk structs with metadata
        for (chunk_id, (chunk_text, keywords)) in doc_chunks.into_iter().enumerate() {
            all_chunks.push(Chunk {
                doc_id,
                chunk_id,
                category: doc_category.clone(),
                text: chunk_text,
                keywords,
            });
        }
    }

    Ok(all_chunks)
}
