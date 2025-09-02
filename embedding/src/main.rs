mod embeddings;

use embeddings::SentenceEmbedder;
use ndarray::Array1;
use std::error::Error;

/// Compute cosine similarity between two vectors.
/// 1.0 means identical direction, 0.0 means orthogonal.
fn cosine_similarity(vec_a: &Array1<f32>, vec_b: &Array1<f32>) -> f32 {
    let dot = vec_a.dot(vec_b);
    let norm_a = vec_a.dot(vec_a).sqrt();
    let norm_b = vec_b.dot(vec_b).sqrt();
    dot / (norm_a * norm_b)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let embedder = SentenceEmbedder::new().await?;

    // Example sentences with similar or related meaning
    let sentences = vec![
        "RAG stands for Retrieval Augmented Generation.",
        "A Large Language Model is a Generative AI model for text generation.",
        "RAG enhance text generation of LLMs by incorporating external data",
        // Added sentence that partially overlaps (relates to LLMs and text generation)
        "Large Language Models can generate coherent text based on training data.",
        // Added sentence on a completely different topic
        "The Pacific Ocean is the largest and deepest ocean on Earth.",
    ];

    // TODO: Add two new sentences here:
    // 1) one that partially overlaps with the first three
    // 2) one that is completely different in topic

    let sentence_refs: Vec<&str> = sentences.iter().map(|s| *s).collect();
    let embeddings = embedder.embed_texts(&sentence_refs).await?;

    // Compare each sentence's embedding to every other using cosine similarity
    for i in 0..embeddings.len() {
        let vec_i = Array1::from(embeddings[i].clone());
        for j in (i + 1)..embeddings.len() {
            let vec_j = Array1::from(embeddings[j].clone());
            let sim_score = cosine_similarity(&vec_i, &vec_j);
            println!(
                "Similarity(\"{}\" , \"{}\") = {:.4}",
                sentences[i], sentences[j], sim_score
            );
        }
    }

    Ok(())
}
