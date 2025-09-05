use regex::Regex;
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
    content: String,
    category: Option<String>,
}

/// Splits the given text into chunks of size 'chunk_size' words, preserving sentence boundaries.
pub fn chunk_text(text: &str, chunk_size: usize) -> Vec<String> {
    // Create regex for splitting on sentence-ending punctuation
    let re = Regex::new(r"(.*?[.!?])\s+").unwrap();

    // Split text into sentences
    let mut sentences: Vec<String> = Vec::new();
    let mut last_end = 0;

    // Collect sentences using regex matches
    for mat in re.find_iter(text) {
        sentences.push(mat.as_str().trim().to_string());
        last_end = mat.end();
    }

    // Add any remaining text as the last sentence (if it doesn't end with punctuation)
    if last_end < text.len() {
        let remaining = text[last_end..].trim();
        if !remaining.is_empty() {
            sentences.push(remaining.to_string());
        }
    }

    // If no sentences were found, treat the entire text as one sentence
    if sentences.is_empty() && !text.trim().is_empty() {
        sentences.push(text.trim().to_string());
    }

    // Group sentences into chunks respecting chunk_size
    let mut chunks: Vec<String> = Vec::new();
    let mut current_chunk = String::new();
    let mut current_word_count = 0;

    for sentence in sentences {
        // Count words in the sentence
        let word_count = sentence.split_whitespace().count();

        // If adding this sentence exceeds chunk_size, start a new chunk
        if current_word_count + word_count > chunk_size && !current_chunk.is_empty() {
            chunks.push(current_chunk.trim().to_string());
            current_chunk = String::new();
            current_word_count = 0;
        }

        // Add sentence to current chunk
        if !current_chunk.is_empty() {
            current_chunk.push(' ');
        }
        current_chunk.push_str(&sentence);
        current_word_count += word_count;
    }

    // Add the last chunk if it contains text
    if !current_chunk.is_empty() {
        chunks.push(current_chunk.trim().to_string());
    }

    chunks
}

/// Loads a dataset from JSON file_path, then splits each document into smaller chunks.
pub fn load_and_chunk_dataset(
    file_path: &str,
    chunk_size: usize,
) -> Result<Vec<Chunk>, Box<dyn Error>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let documents: Vec<Document> = serde_json::from_reader(reader)?;

    let mut all_chunks = Vec::new();

    for (doc_id, doc) in documents.iter().enumerate() {
        let doc_category = doc
            .category
            .clone()
            .unwrap_or_else(|| "general".to_string());
        let doc_chunks = chunk_text(&doc.content, chunk_size);

        for (chunk_id, chunk_str) in doc_chunks.into_iter().enumerate() {
            all_chunks.push(Chunk {
                doc_id,
                chunk_id,
                category: doc_category.clone(),
                text: chunk_str,
            });
        }
    }

    Ok(all_chunks)
}
