mod embeddings;

use embeddings::SentenceEmbedder;
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Load a pre-trained embedding model using our SentenceEmbedder.
    let embedder = SentenceEmbedder::new().await?;

    // Define your own list of three sentences
    let sentences = vec![
        "The sun sets slowly behind the mountain.",
        "A quick fox jumped over the lazy dog.",
        "Stars twinkle brightly in the night sky."
    ];

    // Convert them to a Vec<&str> for embedding
    let sentences_ref: Vec<&str> = sentences.clone(); // Direct assignment since sentences are already &str

    // Encode each sentence into its embedding vector
    let embeddings = embedder.embed_texts(&sentences_ref)?;

    // Print the shape of the embeddings and the first embedding vector
    println!(
        "Embeddings shape: {} sentences, {} dimensions",
        embeddings.len(),
        embeddings.first().map_or(0, |v| v.len())
    );
    println!("First sentence embedding ({}): {:?}", sentences[0], embeddings[0]);

    Ok(())
}
