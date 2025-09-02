use async_openai::{Client, config::OpenAIConfig};
use async_openai::types::CreateEmbeddingRequestArgs;
use std::error::Error;
use std::env;
use dotenv::dotenv;

pub struct SentenceEmbedder {
    client: Client<OpenAIConfig>,
}

impl SentenceEmbedder {
    pub async fn new() -> Result<Self, Box<dyn Error>> {
        dotenv().ok();
        
        println!("Loading OpenAI embedding model (text-embedding-3-small)...");
        let api_key = env::var("OPENAI_API_KEY")
            .expect("OPENAI_API_KEY must be set in .env file");
        
        let config = OpenAIConfig::new().with_api_key(api_key);
        let client = Client::with_config(config);
        
        Ok(Self { client })
    }
    
    pub async fn embed_texts(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
        println!("Embedding {} texts using OpenAI API", texts.len());
        
        let request = CreateEmbeddingRequestArgs::default()
            .model("text-embedding-3-small")
            .input(texts.to_vec())
            .build()?;
            
        let response = self.client.embeddings().create(request).await?;
        
        let embeddings: Vec<Vec<f32>> = response.data
            .into_iter()
            .map(|embedding| embedding.embedding)
            .collect();
            
        println!("Successfully created {} embeddings of dimension {}", 
                embeddings.len(), 
                embeddings.first().map_or(0, |v| v.len()));
        Ok(embeddings)
    }
}
