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

    /// Generate an answer given a query and retrieved context, under different prompting strategies.
    pub async fn generate_with_constraints(
        &self,
        query: &str,
        retrieved_context: &str,
        strategy: &str,
    ) -> Result<(String, String), Box<dyn std::error::Error>> {
        // Fallback if no context
        if retrieved_context.trim().is_empty() {
            return Ok((
                "I'm sorry, but I couldn't find any relevant information.".to_string(),
                "No context used.".to_string(),
            ));
        }

        // Build the prompt according to the chosen strategy
        let prompt = match strategy {
            "strict" => format!(
                "You must ONLY use the context provided below. \
                If you cannot find the answer in the context, say: 'No sufficient data'.\n\
                Do not provide any information not found in the context.\n\n\
                Context:\n{}\n\
                Question: '{}'\n\
                Answer:",
                retrieved_context, query
            ),
            "cite" => format!(
                "Answer the question strictly using the provided context. \
                You must include a 'Cited lines:' section listing the specific lines from the context used to form your answer. \
                If the answer cannot be found in the context, respond with 'Not available in the provided context'.\n\n\
                Context:\n{}\n\
                Question: '{}'\n\
                Answer:",
                retrieved_context, query
            ),
            _ => format!(
                "Use the following context to answer the question in a concise manner.\n\n\
                Context:\n{}\n\
                Question: '{}'\n\
                Answer:",
                retrieved_context, query
            ),
        };

        println!("Prompt:\n{}\n", prompt);

        // Call the LLM
        let response = self.get_llm_response(&prompt).await?;

        // Parse out "Cited lines:" if present
        let parts: Vec<&str> = response.splitn(2, "Cited lines:").collect();
        if parts.len() == 2 {
            let answer = parts[0].trim().to_string();
            let cited = parts[1].trim().to_string();
            Ok((answer, cited))
        } else {
            Ok((response.trim().to_string(), "No explicit lines cited.".to_string()))
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
