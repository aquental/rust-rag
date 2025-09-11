use crate::data::Chunk;
use crate::embeddings::SentenceEmbedder;
use bm25::{Embedder, EmbedderBuilder, Embedding, Language, TokenEmbedding};
use chromadb::collection::QueryOptions;
use std::collections::HashMap;
use std::error::Error;

/// A BM25 “index” that precomputes sparse embeddings for every chunk.
pub struct Bm25Index {
    embedder: Embedder,
    doc_embeddings: Vec<Embedding>,
}

impl Bm25Index {
    /// Build the index by fitting BM25 to the full corpus of chunk texts.
    ///
    /// The returned index precomputes sparse embeddings for every chunk,
    /// which can then be used for BM25 scoring with the `score` method.
    pub fn new(chunks: &[Chunk]) -> Self {
        // Process each chunk: convert to lowercase and collect as owned Strings
        let corpus: Vec<String> = chunks.iter().map(|c| c.text.to_lowercase()).collect();

        // Convert to Vec<&str> for BM25 embedder
        let corpus_refs: Vec<&str> = corpus.iter().map(|s| s.as_str()).collect();

        // Build the BM25 embedder using the processed corpus
        let embedder = EmbedderBuilder::with_fit_to_corpus(Language::English, &corpus_refs).build();

        // Precompute sparse embeddings for each chunk
        let doc_embeddings = corpus_refs
            .iter()
            .map(|&text| embedder.embed(text))
            .collect();

        // Return the Bm25Index
        Bm25Index {
            embedder,
            doc_embeddings,
        }
    }

    /// Compute a BM25‐style score for the query against every chunk.
    ///
    /// The scores are computed as the dot product of the query embedding and
    /// the precomputed sparse embeddings for each chunk.
    pub fn score(&self, query: &str) -> Vec<f32> {
        let q_emb = self.embedder.embed(query);
        self.doc_embeddings
            .iter()
            .map(|doc_emb| dot(&q_emb, doc_emb))
            .collect()
    }
}

/// Dot‐product of two sparse embeddings.
fn dot(a: &Embedding, b: &Embedding) -> f32 {
    let mut sum = 0.0;
    for TokenEmbedding {
        index: qi,
        value: qv,
    } in &a.0
    {
        for TokenEmbedding {
            index: di,
            value: dv,
        } in &b.0
        {
            if qi == di {
                sum += qv * dv;
            }
        }
    }
    sum
}

/// Perform hybrid retrieval combining BM25 scores and dense‐embedding similarity.
///
/// The score for each chunk is a weighted sum of its BM25 score and the
/// cosine similarity between the query embedding and the chunk's dense
/// embedding.  The BM25 scores are normalized to have a range of [0, 1]
/// over the entire corpus, and the dense similarities are also normalized
/// to [0, 1] over the entire corpus.  The final score is a weighted sum of
/// these two normalized scores.
///
/// The function returns a sorted list of (chunk index, score) pairs, with
/// the highest‐scoring pairs first.  The top `top_k` pairs are returned.
pub async fn hybrid_retrieval(
    query: &str,
    chunks: &[Chunk],
    bm25: &Bm25Index,
    collection: &chromadb::collection::ChromaCollection,
    top_k: usize,
    alpha: f32, // weight on BM25 [0..1]
    embedder: &SentenceEmbedder,
) -> Result<Vec<(usize, f32)>, Box<dyn Error>> {
    // 1) BM25 scores + normalization range
    let b_scores = bm25.score(query);
    let (b_min, b_max) = b_scores
        .iter()
        .cloned()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), v| {
            (mn.min(v), mx.max(v))
        });
    let denom = (b_max - b_min).max(f32::EPSILON);

    // 2) Dense retrieval via ChromaDB (we ask only for distances)
    let q_emb = embedder.embed_texts(&[query])?;
    let opts = QueryOptions {
        query_texts: None,
        query_embeddings: Some(q_emb),
        n_results: Some((top_k * 5).min(chunks.len())),
        where_metadata: None,
        where_document: None,
        include: Some(vec!["distances".into()]),
    };
    let res = collection.query(opts, None).await?;

    // 3) Build a map from chunk index → dense similarity
    let mut embed_sim = HashMap::new();
    if let (ids_groups, Some(dist_groups)) = (res.ids, res.distances) {
        if let (Some(ids0), Some(d0)) = (ids_groups.get(0), dist_groups.get(0)) {
            for (i, id_str) in ids0.iter().enumerate() {
                if let Ok(idx) = id_str.parse::<usize>() {
                    let dist = d0.get(i).copied().unwrap_or(0.0);
                    embed_sim.insert(idx, 1.0 / (1.0 + dist));
                }
            }
        }
    }

    // 4) Combine BM25 (normalized) and dense sim into final scores
    let mut merged: Vec<(usize, f32)> = b_scores
        .into_iter()
        .enumerate()
        .map(|(i, b_raw)| {
            let b_norm = (b_raw - b_min) / denom;
            let e_sim = *embed_sim.get(&i).unwrap_or(&0.0);
            (i, alpha * b_norm + (1.0 - alpha) * e_sim)
        })
        .collect();

    // 5) Sort descending and take top_k
    merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    merged.truncate(top_k);

    // 6) Print results
    println!("Top {} hybrid results for '{}':", top_k, query);
    for &(idx, score) in &merged {
        let snippet: String = chunks[idx].text.chars().take(50).collect();
        println!("  Chunk {} (score {:.4}): {}…", idx, score, snippet);
    }

    Ok(merged)
}
