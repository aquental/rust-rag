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
) -> Result<Vec<RetrievedChunk>, Box<dyn std::error::Error>> {
    // Embed the query text to obtain its vector representation
    let query_embedding = embedder.embed_texts(&[query])?;

    // Set up the query options, specifying the use of query embeddings and the number of top results to retrieve
    let query_options = QueryOptions {
        n_results: Some(top_k),
        query_embeddings: Some(query_embedding),
        include: Some(vec!["documents", "distances"]),
        ..Default::default()
    };

    // Execute the query on the collection to retrieve the most relevant document chunks
    let result = collection
        .query(query_options, None)
        .await?;

    let mut chunks = Vec::new();

    // Early return if no results
    if result.documents.is_none() || result.documents.as_ref().unwrap().is_empty() {
        return Ok(chunks);
    }

    let documents = &result.documents.as_ref().unwrap()[0];
    let distances = result.distances.as_ref().map(|d| d[0].clone()).unwrap_or_default();

    for i in 0..documents.len() {
        chunks.push(RetrievedChunk {
            chunk: documents[i].clone(),
            doc_id: i,
            distance: distances.get(i).copied().unwrap_or(0.0),
        });
    }

    Ok(chunks)
}


/// Create (or retrieve) a ChromaDB collection and upsert the full document texts.
pub async fn build_chroma_collection(
    chunks: &[Chunk],
    collection_name: &str,
    embedder: &SentenceEmbedder,
) -> Result<ChromaCollection, Box<dyn std::error::Error>> {
    let client = ChromaClient::new(ChromaClientOptions::default()).await?;
    let collection = client.get_or_create_collection(collection_name, None).await?;

    // Skip empty collection
    if chunks.is_empty() {
        return Ok(collection);
    }

    let texts: Vec<String> = chunks.iter().map(|chunk| chunk.text.clone()).collect();
    let documents: Vec<&str> = texts.iter().map(AsRef::as_ref).collect();

    let ids_owned: Vec<String> = chunks.iter()
        .map(|chunk| format!("doc_{}", chunk.doc_id))
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

    // Embed documents directly
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
