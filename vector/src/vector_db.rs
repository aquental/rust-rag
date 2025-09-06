use chromadb::client::{ChromaClient, ChromaClientOptions};
use chromadb::collection::{ChromaCollection, CollectionEntries};
use serde_json::json;
use crate::data::Chunk;
use crate::embeddings::SentenceEmbedder;

/// Builds or retrieves a ChromaDB collection by embedding each text chunk using the custom SentenceEmbedder.
/// It extracts texts, generates unique IDs and metadata from each chunk, computes embeddings, and upserts
/// the data into the collection.
pub async fn build_chroma_collection(
    chunks: &[Chunk],
    collection_name: &str,
) -> Result<ChromaCollection, Box<dyn std::error::Error>> {

    let embedder = SentenceEmbedder::new().await?;
    
    let client = ChromaClient::new(ChromaClientOptions::default()).await?;
    
    let collection = client.get_or_create_collection(collection_name, None).await?;
    
    let texts: Vec<String> = chunks.iter().map(|chunk| chunk.text.clone()).collect();
    
    let ids_owned: Vec<String> = chunks
        .iter()
        .map(|chunk| format!("chunk_{}_{}", chunk.doc_id, chunk.chunk_id))
        .collect();
    
    let ids: Vec<&str> = ids_owned.iter().map(|s| s.as_str()).collect();
    
    let metadatas: Vec<serde_json::Map<String, serde_json::Value>> = chunks
        .iter()
        .map(|chunk| {
            let mut map = serde_json::Map::new();
            map.insert("doc_id".to_string(), json!(chunk.doc_id));
            map.insert("chunk_id".to_string(), json!(chunk.chunk_id));
            map.insert("category".to_string(), chunk.category.clone().into());
            map
        })
        .collect();
    
    let text_slices: Vec<&str> = texts.iter().map(String::as_str).collect();
    let embeddings = embedder.embed_texts(&text_slices)?;
    
    let documents: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
    
    let entries = CollectionEntries {
        ids,
        embeddings: Some(embeddings),
        metadatas: Some(metadatas),
        documents: Some(documents),
    };
    
    collection.upsert(entries, None).await?;
    
    Ok(collection)
}
