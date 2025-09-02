mod embeddings;

use embeddings::SentenceEmbedder;
use ndarray::Array1;
use std::error::Error;

/// Compute cosine similarity between two vectors.
fn cosine_similarity(vec_a: &Array1<f32>, vec_b: &Array1<f32>) -> f32 {
    let dot = vec_a.dot(vec_b);
    let norm_a = vec_a.dot(vec_a).sqrt();
    let norm_b = vec_b.dot(vec_b).sqrt();
    dot / (norm_a * norm_b)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let embedder = SentenceEmbedder::new().await?;

    // Example sentences
    let sentences = vec![
        "The Eiffel Tower is one of the most famous landmarks in Paris.",
        "Quantum computing promises to revolutionize technology with its speed.",
        "The Amazon rainforest is home to a vast diversity of wildlife.",
        "Meditation can significantly reduce stress and improve mental health.",
        "The Great Wall of China stretches over 13,000 miles.",
    ];

    // Encode each sentence into its embedding vector
    let sentence_refs: Vec<&str> = sentences.iter().map(|s| *s).collect();
    let embeddings = embedder.embed_texts(&sentence_refs)?;

    // Define a query sentence and compute its embedding
    let query = "Famous landmarks attract millions of tourists each year.";
    let query_ref = vec![query];
    let query_embedding = embedder.embed_texts(&query_ref)?[0].clone();

    // Compute similarity of the query to each of the other sentences
    let mut similarities: Vec<(usize, f32, &str)> = Vec::new();
    for (i, embedding) in embeddings.iter().enumerate() {
        let similarity = cosine_similarity(
            &Array1::from(embedding.clone()),
            &Array1::from(query_embedding.clone()),
        );
        similarities.push((i, similarity, sentences[i]));
    }

    // Sort the sentences by similarity score in descending order
    similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Print the sorted sentences with their similarity scores
    println!("Query: '{}'", query);
    println!("Sentences sorted by similarity to query:");
    for (_, similarity, sentence) in similarities {
        println!("{:.4} - {}", similarity, sentence);
    }

    Ok(())
}
