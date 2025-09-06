use crate::data::Chunk;
use crate::embeddings::SentenceEmbedder;
use chromadb::client::{ChromaClient, ChromaClientOptions};
use chromadb::collection::{ChromaCollection, CollectionEntries, GetOptions};
use serde_json::{Map, json};

pub async fn delete_documents_with_keyword(
    collection: &ChromaCollection,
    keyword: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create GetOptions to retrieve all documents from the collection
    let options = GetOptions {
        ids: Vec::new(),
        where_metadata: None,
        limit: None,
        offset: None,
        where_document: None,
        include: Some(vec!["documents".to_string()]),
    };

    // Use collection.get() with the options to retrieve all documents
    let response = collection.get(options).await?;

    // Initialize a vector to store IDs of documents to delete
    let mut ids_to_delete: Vec<String> = Vec::new();

    // If documents are present in the response
    if let Some(documents) = response.documents {
        // Iterate through documents and their indices
        for (index, doc) in documents.iter().enumerate() {
            // Convert keyword to lowercase for case-insensitive comparison
            let keyword_lower = keyword.to_lowercase();
            let doc_lower = doc.as_ref().map(|s| s.to_lowercase()).unwrap_or_default();

            // For each document text that contains the keyword
            if doc_lower.contains(&keyword_lower) {
                // Add its ID to the deletion list
                if let Some(id) = response.ids.get(index) {
                    ids_to_delete.push(id.clone());
                }
            }
        }
    }

    // If there are documents to delete
    if !ids_to_delete.is_empty() {
        // Convert the Vec<String> of IDs to Vec<&str>
        let ids_ref: Vec<&str> = ids_to_delete.iter().map(|s| s.as_str()).collect();

        // Use collection.delete() with the specified parameters
        collection
            .delete(
                Some(ids_ref),
                None, // where_metadata
                None, // where_document
            )
            .await?;
    }

    Ok(())
}

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

    let ids_owned: Vec<String> = chunks
        .iter()
        .map(|chunk| format!("doc_{}_{}", chunk.doc_id, chunk.chunk_id))
        .collect();
    let ids: Vec<&str> = ids_owned.iter().map(AsRef::as_ref).collect();

    let metadatas: Vec<Map<String, serde_json::Value>> = chunks
        .iter()
        .map(|chunk| {
            let mut map = Map::new();
            map.insert("doc_id".to_string(), json!(chunk.doc_id));
            map.insert("chunk_id".to_string(), json!(chunk.chunk_id));
            map.insert("category".to_string(), json!(chunk.category));
            map
        })
        .collect();

    // Embed documents directly
    let embeddings = embedder.embed(&documents).await?;

    let entries = CollectionEntries {
        ids,
        embeddings: Some(embeddings),
        metadatas: Some(metadatas),
        documents: Some(documents),
    };

    collection.upsert(entries, None).await?;
    Ok(collection)
}
