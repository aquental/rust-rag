mod embeddings;

use embeddings::SentenceEmbedder;
use ndarray::{Array1, ArrayView1};
use std::error::Error;

fn cosine_similarity(vec_a: &Array1<f32>, vec_b: &Array1<f32>) -> f32 {
    // Compute dot product
    let dot_product = vec_a.dot(vec_b);
    // Compute L2 norms (magnitudes) of both vectors
    let norm_a = vec_a.dot(vec_a).sqrt();
    let norm_b = vec_b.dot(vec_b).sqrt();
    // Handle division by zero (if either norm is zero, return 0.0)
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        // Cosine similarity = dot product / (norm_a * norm_b)
        dot_product / (norm_a * norm_b)
    }
}

fn embedding_search(
    query: &str,
    docs: &[&str],
    embedder: &SentenceEmbedder,
) -> Result<Vec<(usize, f32)>, Box<dyn Error>> {
    // Encode the query into an embedding vector
    let query_embedding = embedder.encode(query).await?;

    // Encode documents into embedding vectors and compute similarities
    let mut scores = Vec::new();
    for (i, doc) in docs.iter().enumerate() {
        // Encode document into embedding vector
        let doc_embedding = embedder.encode(doc).await?;
        // Compute cosine similarity between query and document embeddings
        let similarity = cosine_similarity(&query_embedding, &doc_embedding);
        scores.push((i, similarity));
    }

    // Sort results by similarity in descending order
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    Ok(scores)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let knowledge_base = [
        "Retrieval-Augmented Generation (RAG) enhances language models by integrating relevant external documents into the generation process.",
        "RAG systems retrieve information from large databases to provide contextual answers beyond what is stored in the model.",
        "By merging retrieved text with generative models, RAG overcomes the limitations of static training data.",
        "Media companies combine external data feeds with digital editing tools to optimize broadcast schedules.",
        "Financial institutions analyze market data and use automated report generation to guide investment decisions.",
        "Healthcare analytics platforms integrate patient records with predictive models to generate personalized care plans.",
        "Bananas are popular fruits that are rich in essential nutrients such as potassium and vitamin C.",
    ];

    let query =
        "How does a system combine external data with language generation to improve responses?";
    println!("Query: {query}");

    let embedder = SentenceEmbedder::new().await?;

    let emb_results = embedding_search(query, &knowledge_base, &embedder)?;
    println!("\nEmbedding-based Search Results:");
    for (idx, score) in emb_results {
        println!(
            "  Doc {idx} | Score: {:.4} | Text: {}",
            score, knowledge_base[idx]
        );
    }

    Ok(())
}
