use serde_json::Value;
use crate::vector_db::retrieve_best_chunk;
use crate::embeddings::SentenceEmbedder;
use chromadb::collection::ChromaCollection;

/// A small set of English stopwords.
const STOPWORDS: &[&str] = &[
    "the","and","is","in","of","to","a","that","for","on","with","as","it","by",
    "this","are","was","at","from","or","be","which","not","can","also","have",
    "has","had","we","they","you","he","she","his","her","its","our","us","their",
    "them","i","do","does","did","just","so","if","may","will","shall","more","most",
    "some","many","any","all","what","about","would","could","should","where","when",
    "why","how"
];

/// Extract up to two non-stopword, non-query words of length > 4 from `chunk_text`.
pub fn extract_refinement_keywords(chunk_text: &str, current_query: &str) -> Vec<String> {
    let chunk_lower = chunk_text.to_lowercase();
    let query_lower = current_query.to_lowercase();

    let chunk_words: Vec<String> = chunk_lower
        .split_whitespace()
        .map(|w| w.chars().filter(|c| c.is_alphanumeric()).collect())
        .filter(|w: &String| !w.is_empty())
        .collect();

    let query_words: std::collections::HashSet<String> = query_lower
        .split_whitespace()
        .map(|w| w.chars().filter(|c| c.is_alphanumeric()).collect())
        .filter(|w: &String| !w.is_empty())
        .collect();

    let mut candidates: Vec<String> = chunk_words.into_iter()
        .filter(|w| w.len() > 4 && !STOPWORDS.contains(&w.as_str()) && !query_words.contains(w))
        .collect();

    // Sort by length in descending order and take up to two keywords
    candidates.sort_by(|a, b| b.len().cmp(&a.len()));
    candidates.into_iter().take(2).collect()
}

/// Append multiple keywords to the `current_query`, if non-empty.
pub fn refine_query(current_query: &str, refine_words: &[String]) -> String {
    if refine_words.is_empty() {
        current_query.to_string()
    } else {
        let keywords = refine_words.join(" ");
        format!("{} {}", current_query, keywords)
    }
}

/// Structure to hold one iteration’s data.
pub struct IterationResult {
    pub step: usize,
    pub query: String,
    pub retrieved_text: String,
    pub metadata: Value,
    pub score: f32,
}

/// Perform up to `steps` rounds of retrieve→extract keywords→refine.
/// Stops if the number of retrieved chunks reaches `max_chunks`.
pub async fn iterative_retrieval(
    collection: &ChromaCollection,
    embedder: &SentenceEmbedder,
    initial_query: &str,
    steps: usize,
    improvement_threshold: f32,
    max_chunks: usize,
) -> Result<Vec<IterationResult>, Box<dyn std::error::Error>> {
    let mut results = Vec::new();
    let mut current_query = initial_query.to_string();
    let mut best_score = 0.0;

    for step in 1..=steps {
        println!("Iteration {}, current query: '{}'", step, current_query);
        let opt = retrieve_best_chunk(collection, embedder, &current_query, 1).await?;
        let (text, score, metadata) = match opt {
            Some(t) => t,
            None => {
                println!("No chunks found at this step. Ending.");
                break;
            }
        };

        println!("Best chunk (50 chars): '{}' | Score: {:.4}", &text[..text.len().min(50)], score);

        if score - best_score < improvement_threshold {
            println!("Improvement threshold not met. Stopping.");
            break;
        }
        best_score = score;

        results.push(IterationResult {
            step,
            query: current_query.clone(),
            retrieved_text: text.clone(),
            metadata,
            score,
        });

        // Stop if we've hit the max_chunks limit
        if results.len() >= max_chunks {
            println!("Reached maximum number of chunks ({}). Stopping.", max_chunks);
            break;
        }

        let keywords = extract_refinement_keywords(&text, &current_query);
        if keywords.is_empty() {
            println!("No suitable keywords for further refinement.");
            break;
        }
        println!("Refining query with keywords: {:?}", keywords);
        current_query = refine_query(&current_query, &keywords);
    }

    Ok(results)
}

/// Combine all iteration texts into one bullet-list context.
pub fn build_final_context(results: &[IterationResult]) -> String {
    if results.is_empty() {
        return "No relevant information was found after iterative retrieval.".to_string();
    }
    results.iter()
        .map(|r| format!("- Step {} (Score={:.4}): {}", r.step, r.score, r.retrieved_text))
        .collect::<Vec<_>>()
        .join("\n")
}
