use crate::data::Chunk;
use crate::embeddings::SentenceEmbedder;
use chromadb::client::{ChromaClient, ChromaClientOptions};
use chromadb::collection::{ChromaCollection, CollectionEntries, QueryOptions};
use serde_json::json;

/// Helper: Convert ISO 8601 string "YYYY-MM-DDTHH:MM:SS" to Unix timestamp (seconds since epoch)
pub fn iso8601_to_timestamp(date_str: &str) -> Option<i64> {
    let parts: Vec<&str> = date_str.split(['T', '-', ':']).collect();
    if parts.len() != 6 {
        return None;
    }
    let year: i32 = parts[0].parse().ok()?;
    let month: u32 = parts[1].parse().ok()?;
    let day: u32 = parts[2].parse().ok()?;
    let hour: u32 = parts[3].parse().ok()?;
    let min: u32 = parts[4].parse().ok()?;
    let sec: u32 = parts[5].parse().ok()?;

    // Days since epoch (ignoring leap seconds, but handling leap years)
    let y = year as i64;
    let m = month as i64;
    let d = day as i64;
    let days = (y - 1970) * 365 + ((y - 1969) / 4) - ((y - 1901) / 100)
        + ((y - 1601) / 400)
        + match m {
            1 => 0,
            2 => 31,
            3 => 59,
            4 => 90,
            5 => 120,
            6 => 151,
            7 => 181,
            8 => 212,
            9 => 243,
            10 => 273,
            11 => 304,
            12 => 334,
            _ => return None,
        }
        + d
        - 1;
    Some(days * 86400 + hour as i64 * 3600 + min as i64 * 60 + sec as i64)
}

/// Helper: Convert Unix timestamp (seconds since epoch) to ISO 8601 string "YYYY-MM-DDTHH:MM:SS"
pub fn timestamp_to_iso8601(ts: i64) -> String {
    // This is a simple implementation for demonstration; for real-world use, prefer a time crate.
    // We'll use UTC.
    let mut s = String::new();
    let mut seconds = ts;
    let days = seconds / 86400;
    seconds -= days * 86400;
    let hour = seconds / 3600;
    seconds -= hour * 3600;
    let min = seconds / 60;
    let sec = seconds - min * 60;

    // Calculate date (naive, not handling all edge cases)
    let mut y = 1970;
    let mut d = days;
    loop {
        let leap = if (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0) {
            366
        } else {
            365
        };
        if d >= leap {
            d -= leap;
            y += 1;
        } else {
            break;
        }
    }
    let month_days = [
        31,
        if (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0) {
            29
        } else {
            28
        },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut m = 1;
    for md in &month_days {
        if d + 1 > *md {
            d -= *md as i64;
            m += 1;
        } else {
            break;
        }
    }
    let day = d + 1;
    s.push_str(&format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
        y, m, day, hour, min, sec
    ));
    s
}

pub struct RetrievedChunk {
    pub chunk: String,
    pub doc_id: usize,
    pub distance: f32,
    pub category: Option<String>,
    pub date: Option<String>,
}

// TODO: Update the function signature to accept min_date parameter
pub async fn metadata_enhanced_search(
    collection: &ChromaCollection,
    query: &str,
    categories: Option<Vec<String>>,
    min_date: Option<&str>,
    top_k: usize,
    embedder: &SentenceEmbedder,
) -> Result<Vec<RetrievedChunk>, Box<dyn std::error::Error>> {
    let query_embedding = embedder.embed_texts(&[query])?;

    // Convert min_date to timestamp if provided
    let min_date_timestamp = min_date.and_then(|date_str| iso8601_to_timestamp(date_str));

    // Build where_clause handling all filter combinations
    let where_clause = match (categories, min_date_timestamp) {
        (Some(cats), Some(timestamp)) => Some(
            json!({
                "$and": [
                    { "category": { "$in": cats } },
                    { "date": { "$gte": timestamp } }
                ]
            })
        ),
        (Some(cats), None) => Some(
            json!({
                "category": { "$in": cats }
            })
        ),
        (None, Some(timestamp)) => Some(
            json!({
                "date": { "$gte": timestamp }
            })
        ),
        (None, None) => None,
    };

    let query_options = QueryOptions {
        query_texts: None,
        query_embeddings: Some(query_embedding),
        n_results: Some(top_k),
        where_metadata: where_clause,
        where_document: None,
        include: Some(vec!["documents", "distances", "metadatas"]),
    };

    let result = collection.query(query_options, None).await?;

    // Create empty vectors as fallbacks
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
            date: metadatas
                .get(i)
                .and_then(|m| m.as_ref())
                .and_then(|m| m.get("date"))
                .and_then(|v| v.as_i64())
                .map(timestamp_to_iso8601),
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

            // Add date to metadata
            if let Some(date_str) = &chunk.date {
                // Parse the date and convert to timestamp
                if let Some(timestamp) = iso8601_to_timestamp(&format!("{}T00:00:00", date_str)) {
                    map.insert("date".to_string(), json!(timestamp));
                }
            }

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
