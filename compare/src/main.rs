use ndarray::Array1;
use std::collections::{HashMap, HashSet};

fn build_vocab(docs: &[&str]) -> HashMap<String, usize> {
    let mut unique_words = HashSet::new();
    for doc in docs {
        for word in doc.to_lowercase().split_whitespace() {
            let clean = word.trim_matches(|c: char| ".,!?".contains(c));
            if !clean.is_empty() {
                unique_words.insert(clean.to_string());
            }
        }
    }
    let mut sorted_words: Vec<_> = unique_words.into_iter().collect();
    sorted_words.sort();
    sorted_words
        .into_iter()
        .enumerate()
        .map(|(idx, word)| (word, idx))
        .collect()
}

fn bow_vectorize(text: &str, vocab: &HashMap<String, usize>) -> Array1<usize> {
    let mut vector = Array1::zeros(vocab.len());
    for word in text.to_lowercase().split_whitespace() {
        let clean = word.trim_matches(|c: char| ".,!?".contains(c));
        if let Some(&idx) = vocab.get(clean) {
            vector[idx] += 1;
        }
    }
    vector
}

fn bow_search(query: &str, docs: &[&str], vocab: &HashMap<String, usize>) -> Vec<(usize, usize)> {
    // Transform the query into a BOW vector
    let query_vec = bow_vectorize(query, vocab);

    // Transform each document into a BOW vector and compute scores
    let mut scores = Vec::new();
    for (i, doc) in docs.iter().enumerate() {
        // Transform document into BOW vector
        let doc_vec = bow_vectorize(doc, vocab);
        // Compute dot product between query and document vectors
        let score = query_vec.dot(&doc_vec);
        scores.push((i, score));
    }

    // Sort documents by score in descending order
    scores.sort_by(|a, b| b.1.cmp(&a.1));

    // Return sorted scores
    scores
}

fn main() {
    let knowledge_base = [
        "Retrieval-Augmented Generation (RAG) enhances language models by integrating relevant external documents into the generation process.",
        "RAG systems retrieve information from large databases to provide contextual answers beyond what is stored in the model.",
        "By merging retrieved text with generative models, RAG overcomes the limitations of static training data.",
        "Media companies combine external data feeds with digital editing tools to optimize broadcast schedules.",
        "Financial institutions analyze market data and use automated report generation to guide investment decisions.",
        "Healthcare analytics platforms integrate patient records with predictive models to generate personalized care plans.",
        "Bananas are popular fruits that are rich in essential nutrients such as potassium and vitamin C.",
    ];

    let vocab = build_vocab(&knowledge_base);
    let query =
        "How does a system combine external data with language generation to improve responses?";

    println!("Query: {query}");

    let bow_results = bow_search(query, &knowledge_base, &vocab);
    println!("BOW Search Results:");
    for (idx, score) in bow_results {
        println!(
            "  Doc {idx} | Score: {score} | Text: {}",
            knowledge_base[idx]
        );
    }
}
