use async_openai::{Client};
use async_openai::config::OpenAIConfig;
use async_openai::types::{
    CreateChatCompletionRequestArgs, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessage, ChatCompletionRequestUserMessage,
    ChatCompletionRequestSystemMessageContent, ChatCompletionRequestUserMessageContent,
};
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
