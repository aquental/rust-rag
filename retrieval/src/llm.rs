use async_openai::{Client};
use async_openai::config::OpenAIConfig;
use async_openai::types::{
    CreateChatCompletionRequestArgs, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessage, ChatCompletionRequestUserMessage,
    ChatCompletionRequestSystemMessageContent, ChatCompletionRequestUserMessageContent,
};
use dotenv::dotenv;
use std::env;
use crate::vector_db::RetrievedChunk;

pub struct LlmClient {
    client: Client<OpenAIConfig>,
    system_prompt: String,
}

impl LlmClient {
    pub fn new() -> Self {
        dotenv().ok();

        let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
        let mut config = OpenAIConfig::new().with_api_key(api_key);

        if let Ok(base_url) = env::var("OPENAI_BASE_URL") {
            config = config.with_api_base(base_url);
        }

        Self {
            client: Client::with_config(config),
            system_prompt: "You are a helpful AI assistant. You always answer to the user's queries.".to_string(),
        }
    }

    pub fn build_prompt(&self, query: &str, retrieved_chunks: &[RetrievedChunk]) -> String {
        // Initialize a string with the question and a directive to use the provided context
        let mut prompt = format!(
            "You are a helpful assistant. Answer the following question based on the provided context. \
             If the answer cannot be found in the context, say so clearly. \
             Use only the information from the context to formulate your response.\n\n"
        );

        // Add context header
        prompt.push_str("===== CONTEXT =====\n\n");

        // Iterate over each retrieved chunk and append it to the prompt
        for (idx, chunk) in retrieved_chunks.iter().enumerate() {
            prompt.push_str(&format!("--- Document {} (Relevance Score: {:.4}) ---\n", idx + 1, 1.0 - chunk.distance));
            prompt.push_str(&chunk.chunk);
            prompt.push_str("\n\n");
        }

        // Add the question section
        prompt.push_str("===== QUESTION =====\n\n");
        prompt.push_str(query);
        prompt.push_str("\n\n");

        // Append a final directive to indicate where the answer should begin
        prompt.push_str("===== ANSWER =====\n\n");
        prompt.push_str("Based on the context provided above, here is my answer:\n\n");
        
        // Return the constructed prompt
        prompt
    }

    pub async fn get_llm_response(&self, prompt: &str) -> Result<String, Box<dyn std::error::Error>> {
        // Build messages using the default system prompt.
        let system_message = ChatCompletionRequestSystemMessage {
            content: ChatCompletionRequestSystemMessageContent::Text(self.system_prompt.clone()),
            name: None,
        };

        let user_message = ChatCompletionRequestUserMessage {
            content: ChatCompletionRequestUserMessageContent::Text(prompt.to_string()),
            name: None,
        };

        let messages = vec![
            ChatCompletionRequestMessage::System(system_message),
            ChatCompletionRequestMessage::User(user_message),
        ];

        let request = CreateChatCompletionRequestArgs::default()
            .model("gpt-4o-mini")
            .messages(messages)
            .temperature(0.0)
            .max_tokens(500_u32)
            .top_p(1.0)
            .frequency_penalty(0.0)
            .presence_penalty(0.0)
            .build()?;

        let response = self.client.chat().create(request).await?;
        let answer = response
            .choices
            .first()
            .and_then(|choice| choice.message.content.clone())
            .unwrap_or_else(|| "No response".to_string());
        Ok(answer)
    }
}
