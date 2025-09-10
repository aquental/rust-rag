use crate::data::Chunk;
use crate::embeddings::SentenceEmbedder;
use chromadb::client::{ChromaClient, ChromaClientOptions};
use chromadb::collection::{ChromaCollection, CollectionEntries};
use serde_json::json;
use std::error::Error;

/// Create (or retrieve) a ChromaDB collection and upsert the full document texts.
pub async fn build_chroma_collection(
    chunks: &[Chunk],
    collection_name: &str,
    embedder: &SentenceEmbedder,
) -> Result<ChromaCollection, Box<dyn Error>> {
    let client = ChromaClient::new(ChromaClientOptions::default()).await?;
    let collection = client.get_or_create_collection(collection_name, None).await?;

    // Skip empty collection
    if chunks.is_empty() {
        return Ok(collection);
    }

    let texts: Vec<String> = chunks.iter().map(|chunk| chunk.text.clone()).collect();
    let documents: Vec<&str> = texts.iter().map(AsRef::as_ref).collect();

    // Create unique IDs by combining doc_id and chunk_id
    let ids_owned: Vec<String> = chunks.iter()
        .map(|chunk| format!("doc_{}_chunk_{}", chunk.doc_id, chunk.chunk_id))
        .collect();
    let ids: Vec<&str> = ids_owned.iter().map(AsRef::as_ref).collect();

    let metadatas = chunks.iter()
        .map(|chunk| {
            let mut map = serde_json::Map::new();
            map.insert("doc_id".to_string(), json!(chunk.doc_id));
            map.insert("chunk_id".to_string(), json!(chunk.chunk_id));
            map.insert("category".to_string(), chunk.category.clone().into());
            map
        })
        .collect();

    let embeddings = embedder.embed_texts(&documents)?;

    let entries = CollectionEntries {
        ids,
        embeddings: Some(embeddings),
        metadatas: Some(metadatas),
        documents: Some(documents),
    };

    collection.upsert(entries, None).await?;
    Ok(collection)
}
