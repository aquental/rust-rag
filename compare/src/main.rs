mod embeddings;

use embeddings::SentenceEmbedder;
use ndarray::Array1;
use std::error::Error;

fn cosine_similarity(vec_a: &Array1<f32>, vec_b: &Array1<f32>) -> f32 {
    // Calculate dot product
    let dot_product = vec_a.dot(vec_b);

    // Calculate L2 norms (magnitude) of both vectors
    let norm_a = vec_a.dot(vec_a).sqrt();
    let norm_b = vec_b.dot(vec_b).sqrt();

    // Handle division by zero
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    // Calculate cosine similarity: dot_product / (norm_a * norm_b)
    dot_product / (norm_a * norm_b)
}

fn embedding_search(
    query: &str,
    docs: &[&str],
    embedder: &SentenceEmbedder,
) -> Result<Vec<(usize, f32)>, Box<dyn Error>> {
    // Encode the query into an embedding vector
    let query_embedding = embedder.embed_texts(&[query])?;
    let query_embedding = Array1::from_vec(query_embedding[0].clone());

    // Encode all documents into embedding vectors
    let doc_embeddings = embedder.embed_texts(docs)?;

    // Calculate cosine similarity between query and each document
    let mut results: Vec<(usize, f32)> = doc_embeddings
        .iter()
        .enumerate()
        .map(|(idx, doc_emb)| {
            let doc_array = Array1::from_vec(doc_emb.clone());
            let similarity = cosine_similarity(&query_embedding, &doc_array);
            (idx, similarity)
        })
        .collect();

    // Sort results by similarity in descending order
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    Ok(results)
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
