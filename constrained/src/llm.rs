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
    /// TODO: Add context-length validation and smart truncation if the context exceeds a limit of 4096 tokens (approx. word-based).
    /// If truncation occurs, "[Context truncated]" should be appended to the answer.
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

        // Approximate token limit
        const MAX_TOKENS: usize = 4096;

        // Check and truncate context if too large
        let mut context = retrieved_context.to_string();
        let mut truncated = false;

        // Approximate token count (1 token ≈ 0.75 words)
        let word_count = context.split_whitespace().count();
        let approx_tokens = (word_count as f32 / 0.75).ceil() as usize;

        if approx_tokens > MAX_TOKENS {
            truncated = true;
            // Split context into sentences
            let sentences: Vec<&str> = context
                .split_inclusive(&['.', '!', '?'])
                .filter(|s| !s.trim().is_empty())
                .collect();
            let mut truncated_context = String::new();
            let mut current_tokens = 0;

            // Add sentences until reaching token limit
            for sentence in sentences {
                let sentence_words = sentence.split_whitespace().count();
                let sentence_tokens = (sentence_words as f32 / 0.75).ceil() as usize;
                if current_tokens + sentence_tokens <= MAX_TOKENS {
                    truncated_context.push_str(sentence);
                    current_tokens += sentence_tokens;
                } else {
                    break;
                }
            }

            context = truncated_context;
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
                context, query
            ),
            "cite" => format!(
                "Answer strictly from the provided context, and list the lines you used as evidence with 'Cited lines:'.\
                If the context does not contain the information, respond with: 'Not available in the retrieved texts.'\n\n\
                Provided context (label lines as needed):\n{}\n\
                Question: '{}'\n\
                Answer:",
                context, query
            ),
            _ => format!(
                "Use the following context to answer the question in a concise manner.\n\n\
                Context:\n{}\n\
                Question: '{}'\n\
                Answer:",
                context, query
            ),
        };

        println!("Prompt:\n{}\n", prompt);

        // Call the LLM
        let response = self.get_llm_response(&prompt).await?;

        // Parse out "Cited lines:" if present
        let parts: Vec<&str> = response.splitn(2, "Cited lines:").collect();
        let (mut answer, cited) = if parts.len() == 2 {
            (parts[0].trim().to_string(), parts[1].trim().to_string())
        } else {
            (response.trim().to_string(), "No explicit lines cited.".to_string())
        };

        // Append truncation warning if context was truncated
        if truncated {
            answer.push_str(" [Context truncated]");
        }

        Ok((answer, cited))
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
