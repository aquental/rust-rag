mod llm;

use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
struct Document {
    title: String,
    content: String,
}

type KnowledgeBase = HashMap<String, Document>;

fn create_knowledge_base() -> KnowledgeBase {
    let mut kb = HashMap::new();

    kb.insert(
        "AAPL".to_string(),
        Document {
            title: "AAPL Stock (April 2023)".to_string(),
            content: "On 2023-04-13, AAPL opened at $160.50, closed at $162.30, \
                     with a high of $163.00 and a low of $159.90. \
                     Trading volume was 80 million shares. \
                     On 2023-04-14, AAPL opened at $161.10, closed at $162.80, \
                     with a high of $163.50 and a low of $160.50. \
                     Trading volume was 85 million shares."
                .to_string(),
        },
    );

    kb.insert(
        "MSFT".to_string(),
        Document {
            title: "MSFT Stock (April 2023)".to_string(),
            content: "On 2023-04-13, MSFT opened at $285.00, closed at $288.50, \
                     with a high of $290.00 and a low of $283.50. \
                     Trading volume was 35 million shares. \
                     On 2023-04-14, MSFT opened at $286.00, closed at $289.00, \
                     with a high of $291.50 and a low of $284.70. \
                     Trading volume was 40 million shares."
                .to_string(),
        },
    );

    kb.insert(
        "TSLA".to_string(),
        Document {
            title: "TSLA Stock (April 2023)".to_string(),
            content: "On 2023-04-13, TSLA opened at $185.00, closed at $187.00, \
                     with a high of $189.00 and a low of $184.50. \
                     Trading volume was 50 million shares. \
                     On 2023-04-14, TSLA opened at $186.00, closed at $188.50, \
                     with a high of $190.00 and a low of $185.50. \
                     Trading volume was 55 million shares."
                .to_string(),
        },
    );

    kb
}

/// Naively generate a response for the query.
async fn naive_generation(
    query: &str,
    llm: &llm::LlmClient,
) -> Result<String, Box<dyn std::error::Error>> {
    let prompt = format!("Answer directly the following query: {}", query);
    llm.get_llm_response(&prompt).await
}

/// Retrieve the document from the knowledge base with highest word overlap.
fn rag_retrieval<'a>(query: &str, documents: &'a KnowledgeBase) -> Option<&'a Document> {
    let query_lower = query.to_lowercase();
    let query_words: HashSet<_> = query_lower.split_whitespace().collect();

    documents
        .values()
        .map(|doc| {
            let doc_lower = doc.content.to_lowercase();
            let doc_words: HashSet<_> = doc_lower.split_whitespace().collect();
            let overlap = query_words.intersection(&doc_words).count();
            (doc, overlap)
        })
        .max_by_key(|(_, overlap)| *overlap)
        .map(|(doc, _)| doc)
}

/// Generate a response using the retrieved document as context.
async fn rag_generation(
    query: &str,
    document: Option<&Document>,
    llm: &llm::LlmClient,
) -> Result<String, Box<dyn std::error::Error>> {
    // Extract requested stock symbols from the query
    let query_lower = query.to_lowercase();
    let query_words: HashSet<String> = query_lower
        .split_whitespace()
        .map(|word| word.trim_matches(|c| c == ',' || c == '.').to_string())
        .collect();
    let stock_symbols: Vec<String> = query_words
        .into_iter()
        .filter(|word| word.chars().all(|c| c.is_alphabetic()))
        .collect();

    // Prepare the prompt based on document availability and completeness
    let prompt = match document {
        Some(doc) => {
            // Check if the document contains data for all requested symbols
            let doc_lower = doc.title.to_lowercase();
            let has_all_symbols = stock_symbols
                .iter()
                .all(|symbol| doc_lower.contains(&symbol.to_lowercase()));
            if has_all_symbols {
                format!(
                    "Using the following information: '{}: {}', provide a confident and accurate answer to the query: '{}'",
                    doc.title, doc.content, query
                )
            } else {
                format!(
                    "The available information: '{}: {}' does not contain sufficient data for all requested stock symbols. \
                    Politely refuse to answer the query, stating that there isn't enough information to respond accurately: '{}'",
                    doc.title, doc.content, query
                )
            }
        }
        None => {
            format!(
                "No relevant information was found in the knowledge base for the requested stock symbols. \
                Politely refuse to answer the query, stating that there isn't enough information to respond accurately: '{}'",
                query
            )
        }
    };

    llm.get_llm_response(&prompt).await
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let kb = create_knowledge_base();

    let query = "Write a short summary of the stock market performance on April 14, \
                 2023 for the following symbols: NVDA, GOOG.\n\
                 Your summary should include:\n\
                 For each symbol:\n\
                 - The opening price\n\
                 - The closing price\n\
                 - The highest and lowest prices of the day\n\
                 - The trading volume";

    let llm_client = llm::LlmClient::new();

    println!(
        "Naive approach:\n{}",
        naive_generation(query, &llm_client).await?
    );

    let retrieved_doc = rag_retrieval(query, &kb);
    println!(
        "\n\nRAG approach:\n{}",
        rag_generation(query, retrieved_doc, &llm_client).await?
    );

    Ok(())
}
