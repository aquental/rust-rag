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

/// Loads the dataset from the given JSON file and returns full documents as single chunks.
pub fn load_documents(file_path: &str) -> Result<Vec<Chunk>, Box<dyn Error>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let documents: Vec<Document> = serde_json::from_reader(reader)?;

    // Use map to convert each Document into a Chunk.
    let docs = documents
        .into_iter()
        .map(|doc| {
            let category = doc.category.unwrap_or_else(|| "general".to_string());
            Chunk {
                doc_id: doc.id,
                chunk_id: 0,
                category,
                text: doc.content,
            }
        })
        .collect();

    Ok(docs)
}
