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

/// Splits the given text into chunks of size 'chunk_size' with specified word overlap, preserving sentence boundaries.
pub fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    // Handle edge cases
    if chunk_size == 0 {
        return vec![];
    }
    if overlap >= chunk_size {
        return vec![text.to_string()];
    }

    // Split text into sentences using regex
    let re = Regex::new(r"(.*?[.!?])\s+").unwrap();
    let mut sentences: Vec<String> = Vec::new();
    let mut last_end = 0;

    for mat in re.find_iter(text) {
        sentences.push(mat.as_str().trim().to_string());
        last_end = mat.end();
    }
    if last_end < text.len() {
        let remaining = text[last_end..].trim();
        if !remaining.is_empty() {
            sentences.push(remaining.to_string());
        }
    }
    if sentences.is_empty() && !text.trim().is_empty() {
        sentences.push(text.trim().to_string());
    }

    // Split sentences into words
    let words: Vec<&str> = text.split_whitespace().filter(|w| !w.is_empty()).collect();
    if words.is_empty() {
        return vec![];
    }

    // Compute step size
    let step = chunk_size.saturating_sub(overlap);

    // Group words into chunks with overlap
    let mut chunks: Vec<String> = Vec::new();
    let mut sentence_index = 0;
    let mut word_count = 0;
    let mut current_chunk = String::new();
    let mut current_words: Vec<&str> = Vec::new();

    for i in (0..words.len()).step_by(step) {
        // Collect words for the current chunk
        let end = std::cmp::min(i + chunk_size, words.len());
        let chunk_words = &words[i..end];

        // Find sentences that cover these words
        current_chunk.clear();
        while sentence_index < sentences.len() {
            let sentence = &sentences[sentence_index];
            let sentence_words = sentence.split_whitespace().count();

            // Check if adding this sentence exceeds chunk_size
            if word_count + sentence_words > chunk_size && !current_chunk.is_empty() {
                break;
            }

            // Add sentence to chunk
            if !current_chunk.is_empty() {
                current_chunk.push(' ');
            }
            current_chunk.push_str(sentence);
            word_count += sentence_words;
            sentence_index += 1;
        }

        // If we have content, add the chunk
        if !current_chunk.is_empty() {
            chunks.push(current_chunk.clone());
            word_count = 0; // Reset word count for next chunk
        }

        // If we've reached the end of sentences, break
        if sentence_index >= sentences.len() && end >= words.len() {
            break;
        }

        // Move sentence_index back to include overlap
        if overlap > 0 {
            let mut overlap_words = 0;
            let mut temp_index = sentence_index;
            while temp_index > 0 && overlap_words < overlap {
                temp_index -= 1;
                overlap_words += sentences[temp_index].split_whitespace().count();
            }
            sentence_index = temp_index;
            word_count = overlap_words;
            current_chunk = sentences[temp_index..sentence_index].join(" ");
            if !current_chunk.is_empty() {
                current_chunk.push(' ');
            }
        } else {
            word_count = 0;
            current_chunk.clear();
        }
    }

    // Add any remaining content
    if !current_chunk.is_empty() && sentence_index < sentences.len() {
        while sentence_index < sentences.len() {
            if !current_chunk.is_empty() {
                current_chunk.push(' ');
            }
            current_chunk.push_str(&sentences[sentence_index]);
            sentence_index += 1;
        }
        chunks.push(current_chunk.trim().to_string());
    }

    chunks
}

/// Loads a dataset from JSON file_path, then splits each document into smaller chunks.
pub fn load_and_chunk_dataset(
    file_path: &str,
    chunk_size: usize,
    overlap: usize,
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
        let doc_chunks = chunk_text(&doc.content, chunk_size, overlap);

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
