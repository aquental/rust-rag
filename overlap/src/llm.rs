use async_openai::config::OpenAIConfig;
use async_openai::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
    ChatCompletionRequestSystemMessageContent, ChatCompletionRequestUserMessage,
    ChatCompletionRequestUserMessageContent, CreateChatCompletionRequestArgs,
};
use async_openai::Client;
use dotenv::dotenv;
use std::env;

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

    /// Generates answers based on context
    ///
    /// Checks if context is empty, then create a prompt with query and context
    /// If context is empty or whitespace-only, returns a default message
    /// Otherwise, formulates the prompt with query and context and generates a response using get_llm_response
    pub async fn generate_final_answer(
        &self,
        query: &str,
        context: &str,
    ) -> Result<String, Box<dyn std::error::Error>> {
        // Handle empty or whitespace-only context
        if context.trim().is_empty() {
            return Ok("I'm sorry, but I couldn't find any relevant information.".to_string());
        }

        // Formulate the prompt with query and context
        let prompt = format!(
            "Question: {}\nContext:\n{}\nAnswer:",
            query, context
        );

        // Generate response using get_llm_response
        let response = self.get_llm_response(&prompt).await?;
        Ok(response)
    }

    /// Generates a response using the given prompt with the LLM client.
    ///
    /// This function takes a prompt string, builds a default system message with the
    /// client's system prompt, and then builds a user message with the given prompt.
    /// It then creates a `CreateChatCompletionRequest` with the two messages, and
    /// calls the `chat().create()` method of the client to generate a response.
    /// The response is then extracted from the result and returned as a string.
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
