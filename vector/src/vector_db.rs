use crate::data::Chunk;
use crate::embeddings::SentenceEmbedder;
use chromadb::client::{ChromaClient, ChromaClientOptions};
use chromadb::collection::{ChromaCollection, CollectionEntries};
use serde_json::{json, Map, Value};

pub async fn build_chroma_collection(
    chunks: &[Chunk],
    collection_name: &str,
    embedder: &SentenceEmbedder,
) -> Result<ChromaCollection, Box<dyn std::error::Error>> {
    // Connect to ChromaClient and create/retrieve collection
    let client = ChromaClient::new(ChromaClientOptions::default()).await?;
    let collection = client
        .get_or_create_collection(collection_name, None)
        .await?;

    // Handle empty chunks case
    if chunks.is_empty() {
        return Ok(collection);
    }

    // Extract texts, ids, and metadata from the chunks
    let texts: Vec<String> = chunks.iter().map(|chunk| chunk.text.clone()).collect();
    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
    let ids: Vec<String> = chunks
        .iter()
        .map(|chunk| format!("{}_{}", chunk.doc_id, chunk.chunk_id))
        .collect();
    let id_refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
    let metadatas: Vec<Map<String, Value>> = chunks
        .iter()
        .map(|chunk| {
            let mut map = Map::new();
            map.insert("doc_id".to_string(), json!(chunk.doc_id));
            map.insert("chunk_id".to_string(), json!(chunk.chunk_id));
            map.insert("category".to_string(), json!(chunk.category));
            map
        })
        .collect();

    // Embed the texts
    let embeddings = embedder.embed_texts(&text_refs)?;

    // Add entries to the collection using collection.upsert
    let entries = CollectionEntries {
        ids: id_refs,
        embeddings: Some(embeddings),
        metadatas: Some(metadatas),
        documents: Some(text_refs),
    };

    collection.upsert(entries, None).await?;

    Ok(collection)
}
