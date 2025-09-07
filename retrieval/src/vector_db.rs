use chromadb::client::{ChromaClient, ChromaClientOptions};
use chromadb::collection::{ChromaCollection, CollectionEntries, QueryOptions};
use serde_json::json;
use crate::data::Chunk;
use crate::embeddings::SentenceEmbedder;

pub struct RetrievedChunk {
    pub chunk: String,
    pub doc_id: usize,
    pub distance: f32,
}

pub async fn retrieve_top_chunks(
    collection: &ChromaCollection,
    query: &str,
    top_k: usize,
    embedder: &SentenceEmbedder,
    category_filter: Option<&str>,
    distance_threshold: Option<f32>,
) -> Result<Vec<RetrievedChunk>, Box<dyn std::error::Error>> {

    let query_embeddings = embedder.embed_texts(&[query])?;

    // Build metadata filter if category is provided
    let where_metadata = category_filter.map(|category| {
        json!({"category": category})
    });

    // Request more results than top_k to account for filtering by distance
    let query_n = if distance_threshold.is_some() {
        top_k * 3  // Request more to ensure we have enough after filtering
    } else {
        top_k
    };

    let query_options = QueryOptions {
        query_texts: None,
        query_embeddings: Some(query_embeddings),
        n_results: Some(query_n),
        where_metadata,
        where_document: None,
        include: Some(vec!["documents", "distances", "metadatas"]),
    };

    let query_result = collection.query(query_options, None).await?;
    let mut retrieved_chunks = Vec::new();

    if let Some(documents_groups) = query_result.documents.as_ref() {
        if let Some(documents) = documents_groups.get(0) {
            for (i, doc) in documents.iter().enumerate() {
                let distance = query_result
                    .distances
                    .as_ref()
                    .and_then(|rows| rows.get(0))
                    .and_then(|row| row.get(i))
                    .copied()
                    .unwrap_or(0.0);

                // Apply distance threshold filtering
                // Note: In ChromaDB, lower distance = higher similarity
                // Typical distance ranges: 0.0 (identical) to 2.0 (completely different)
                if let Some(threshold) = distance_threshold {
                    if distance > threshold {
                        continue; // Skip chunks that are too dissimilar
                    }
                }

                // Extract doc_id from metadata if available
                let doc_id = query_result
                    .metadatas
                    .as_ref()
                    .and_then(|rows| rows.get(0))
                    .and_then(|row| row.get(i))
                    .and_then(|metadata| metadata.as_ref())
                    .and_then(|metadata| metadata.get("doc_id"))
                    .and_then(|value| value.as_u64())
                    .map(|id| id as usize)
                    .unwrap_or(i); // Fallback to index if metadata not found

                retrieved_chunks.push(RetrievedChunk {
                    chunk: doc.clone(),
                    doc_id,
                    distance,
                });

                // Stop if we've collected enough chunks
                if retrieved_chunks.len() >= top_k {
                    break;
                }
            }
        }
    }

    Ok(retrieved_chunks)
}


/// Create (or retrieve) a ChromaDB collection and upsert the full document texts.
pub async fn build_chroma_collection(
    chunks: &[Chunk],
    collection_name: &str,
    embedder: &SentenceEmbedder,
) -> Result<ChromaCollection, Box<dyn std::error::Error>> {

    let client = ChromaClient::new(ChromaClientOptions::default()).await?;
    let collection = client.get_or_create_collection(collection_name, None).await?;

    // Use the entire document content.
    let texts: Vec<String> = chunks.iter().map(|chunk| chunk.text.clone()).collect();

    // Create a unique ID for each document
    let ids_owned: Vec<String> = chunks
        .iter()
        .map(|chunk| format!("doc_{}", chunk.doc_id))
        .collect();
    let ids: Vec<&str> = ids_owned.iter().map(|s| s.as_str()).collect();

    // Prepare metadata for each document.
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

    // Get embeddings for the documents.
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
