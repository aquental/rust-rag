use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType,
};
use std::error::Error;

pub struct SentenceEmbedder {
    model: rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModel,
}

impl SentenceEmbedder {
    pub async fn new() -> Result<Self, Box<dyn Error>> {
        println!("Loading sentence embedding model (all-MiniLM-L6-v2)...");
        let model = tokio::task::spawn_blocking(|| {
            SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL6V2)
                .create_model()
        })
        .await??;

        Ok(Self { model })
    }

    pub fn embed_texts(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
        println!("Embedding {} texts", texts.len());
        let embeddings = self.model.encode(texts)?;
        println!(
            "Successfully created {} embeddings of dimension {}",
            embeddings.len(),
            embeddings.first().map_or(0, |v| v.len())
        );
        Ok(embeddings)
    }
}
