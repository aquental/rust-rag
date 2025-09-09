use std::error::Error;
use crate::data::Chunk;
use crate::embeddings::SentenceEmbedder;
use chromadb::client::{ChromaClient, ChromaClientOptions};
use chromadb::collection::{ChromaCollection, CollectionEntries, QueryOptions};
use serde_json::{json, Value};

/// Returns `(chunk_text, inverted_score, metadata)` for the top match, or `None` if no documents.
pub async fn retrieve_best_chunk(
    collection: &ChromaCollection,
    embedder: &SentenceEmbedder,
    query: &str,
    n_results: usize,
) -> Result<Option<(String, f32, Value)>, Box<dyn Error>> {
    let query_embeddings = embedder.embed_texts(&[query])?;
    let opts = QueryOptions {
        query_texts: None,
        query_embeddings: Some(query_embeddings),
        n_results: Some(n_results),
        where_metadata: None,
        where_document: None,
        include: Some(vec!["documents".into(), "distances".into(), "metadatas".into()]),
    };

    let res = collection.query(opts, None).await?;

    // Create a longer-lived empty vector
    let empty_vec = Vec::new();
    let docs = res
        .documents
        .as_ref()
        .and_then(|groups| groups.get(0))
        .unwrap_or(&empty_vec);

    if docs.is_empty() {
        return Ok(None);
    }

    let text = docs[0].clone();
    let distance = res
        .distances
        .as_ref()
        .and_then(|groups| groups.get(0))
        .and_then(|row| row.get(0))
        .copied()
        .unwrap_or(0.0);

    // Inverted-distance similarity score formula.
    // Use 1.0 / (1.0 + distance) to convert distance to similarity.
    let score = 1.0 / (1.0 + distance);

    // Handle metadata conversion
    let metadata = res
        .metadatas
        .as_ref()
        .and_then(|groups| groups.get(0))
        .and_then(|row| row.get(0))
        .and_then(|m| serde_json::to_value(m).ok())
        .unwrap_or(Value::Null);

    Ok(Some((text, score, metadata)))
}

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
