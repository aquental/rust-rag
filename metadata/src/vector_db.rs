use crate::data::Chunk;
use crate::embeddings::SentenceEmbedder;
use chromadb::client::{ChromaClient, ChromaClientOptions};
use chromadb::collection::{ChromaCollection, CollectionEntries, QueryOptions};
use serde_json::json;

pub struct RetrievedChunk {
    pub chunk: String,
    pub doc_id: usize,
    pub distance: f32,
    pub category: Option<String>,
}

pub async fn metadata_enhanced_search(
    collection: &ChromaCollection,
    query: &str,
    categories: Option<Vec<String>>,
    top_k: usize,
    embedder: &SentenceEmbedder,
) -> Result<Vec<RetrievedChunk>, Box<dyn std::error::Error>> {
    // Create query embedding
    let query_embedding = embedder.embed_texts(&[query])?;

    // Create where_clause using as_ref to avoid moving categories
    let where_clause = categories.as_ref().map(|cats| {
        json!({
            "category": { "$in": cats }
        })
        .as_object()
        .cloned()
        .unwrap_or_default()
    });

    // Build initial QueryOptions
    let query_options = QueryOptions {
        query_texts: None,
        query_embeddings: Some(query_embedding.clone()),
        n_results: Some(top_k),
        where_metadata: where_clause.map(serde_json::Value::Object),
        where_document: None,
        include: Some(vec!["documents", "distances", "metadatas"]),
    };

    // Execute initial query
    let mut result = collection.query(query_options, None).await?;

    // Check if initial search returned no results
    if categories.is_some()
        && result
            .documents
            .as_ref()
            .and_then(|d| d.first())
            .map_or(true, |docs| docs.is_empty())
    {
        // Perform fallback search without filter
        let fallback_options = QueryOptions {
            query_texts: None,
            query_embeddings: Some(query_embedding),
            n_results: Some(top_k),
            where_metadata: None,
            where_document: None,
            include: Some(vec!["documents", "distances", "metadatas"]),
        };
        result = collection.query(fallback_options, None).await?;
    }

    // Process results
    let documents = result
        .documents
        .and_then(|d| d.first().cloned())
        .unwrap_or_default();

    let distances = result
        .distances
        .and_then(|d| d.first().cloned())
        .unwrap_or_default();

    let metadatas = result
        .metadatas
        .and_then(|m| m.first().cloned())
        .unwrap_or_default();

    Ok(documents
        .iter()
        .enumerate()
        .map(|(i, chunk)| RetrievedChunk {
            chunk: chunk.clone(),
            doc_id: metadatas
                .get(i)
                .and_then(|m| m.as_ref())
                .and_then(|m| m.get("doc_id"))
                .and_then(|v| v.as_u64())
                .map(|id| id as usize)
                .unwrap_or(i),
            category: metadatas
                .get(i)
                .and_then(|m| m.as_ref())
                .and_then(|m| m.get("category"))
                .and_then(|v| v.as_str())
                .map(String::from),
            distance: distances.get(i).copied().unwrap_or(0.0),
        })
        .collect())
}

/// Create (or retrieve) a ChromaDB collection and upsert the full document texts.
pub async fn build_chroma_collection(
    chunks: &[Chunk],
    collection_name: &str,
    embedder: &SentenceEmbedder,
) -> Result<ChromaCollection, Box<dyn std::error::Error>> {
    let client = ChromaClient::new(ChromaClientOptions::default()).await?;
    let collection = client
        .get_or_create_collection(collection_name, None)
        .await?;

    // Skip empty collection
    if chunks.is_empty() {
        return Ok(collection);
    }

    let texts: Vec<String> = chunks.iter().map(|chunk| chunk.text.clone()).collect();
    let documents: Vec<&str> = texts.iter().map(AsRef::as_ref).collect();

    // Create unique IDs by combining doc_id and chunk_id
    let ids_owned: Vec<String> = chunks
        .iter()
        .map(|chunk| format!("doc_{}_chunk_{}", chunk.doc_id, chunk.chunk_id))
        .collect();
    let ids: Vec<&str> = ids_owned.iter().map(AsRef::as_ref).collect();

    let metadatas = chunks
        .iter()
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
