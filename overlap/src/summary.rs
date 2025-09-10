use crate::llm::LlmClient;
use std::collections::HashSet;


/// Determine if there is significant lexical overlap between the given chunks.
///
/// This function takes a vector of strings, splits each string into words,
/// and checks for significant overlap between any two strings. The overlap
/// is computed as the size of the intersection divided by the size of the
/// larger set. If the overlap ratio is above the given threshold, the
/// function returns true. Otherwise, it returns false.
///
pub fn are_chunks_overlapping(chunks: &[String], similarity_threshold: f32) -> bool {
    if chunks.len() < 2 {
        return false;
    }

    // Step 1: Split each chunk into words and convert to a HashSet<String>
    let word_sets: Vec<HashSet<String>> = chunks
        .iter()
        .map(|chunk| {
            chunk
                .split_whitespace() // Split text into words
                .map(|w| w.to_lowercase()) // Convert to lowercase for case-insensitive comparison
                .collect::<HashSet<String>>() // Collect into a HashSet
        })
        .collect();

    // Step 2: Compare each pair of word sets
    for i in 0..word_sets.len() - 1 {
        for j in i + 1..word_sets.len() {
            // Step 3: Compute the intersection size and the size of the larger set
            let intersection_size = word_sets[i]
                .intersection(&word_sets[j])
                .count() as f32; // Size of common words
            let max_len = word_sets[i]
                .len()
                .max(word_sets[j].len()) as f32; // Size of larger set

            // Step 4: Calculate overlap ratio as intersection / max_len
            let overlap: f32 = if max_len > 0.0 {
                intersection_size / max_len
            } else {
                0.0 // Avoid division by zero if both sets are empty
            };

            // Step 5: Compare with threshold
            if overlap > similarity_threshold {
                return true;
            }
        }
    }
    false
}

/// Summarize the given chunks of text using the LLM.
/// If the summary is shorter than 20 characters or signals that a summary is not possible,
/// return the full text of the chunks instead.
pub async fn summarize_chunks(
    llm: &LlmClient,
    chunks: &[String],
) -> Result<String, Box<dyn std::error::Error>> {
    if chunks.is_empty() {
        return Ok("No relevant chunks were retrieved.".to_string());
    }

    let combined = chunks.join("\n");
    let prompt = format!(
        "You are an expert summarizer. Please generate a concise summary of the following text.\n\
         Do not omit critical details that might answer the user's query.\n\
         If you cannot produce a meaningful summary, just say 'Summary not possible'.\n\n\
         Text:\n{}\n\nSummary:",
        combined
    );

    let summary = llm.get_llm_response(&prompt).await?.trim().to_string();

    if summary.len() < 20 || summary.contains("Summary not possible") {
        eprintln!("Summary was too short or signaled not possible; returning full text.");
        Ok(combined)
    } else {
        Ok(summary)
    }
}
