use ndarray::Array1;
use std::collections::{HashMap, HashSet};

fn build_vocab(docs: &[&str]) -> HashMap<String, usize> {
    let mut unique_tokens = HashSet::new();
    for doc in docs {
        let words: Vec<String> = doc
            .to_lowercase()
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| ".,!?".contains(c)).to_string())
            .filter(|w| !w.is_empty())
            .collect();
        // Add unigrams
        for i in 0..words.len() {
            unique_tokens.insert(words[i].clone());
            // Add bigrams
            if i < words.len() - 1 {
                let bigram = format!("{} {}", words[i], words[i + 1]);
                unique_tokens.insert(bigram);
            }
        }
    }
    let mut sorted_tokens: Vec<_> = unique_tokens.into_iter().collect();
    sorted_tokens.sort();
    sorted_tokens
        .into_iter()
        .enumerate()
        .map(|(i, tok)| (tok, i))
        .collect()
}

fn bow_vectorize(text: &str, vocab: &HashMap<String, usize>) -> Array1<usize> {
    let words: Vec<String> = text
        .to_lowercase()
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| ".,!?".contains(c)).to_string())
        .filter(|w| !w.is_empty())
        .collect();
    let mut vector = Array1::zeros(vocab.len());
    for i in 0..words.len() {
        // Count unigrams
        if let Some(&idx) = vocab.get(&words[i]) {
            vector[idx] += 1;
        }
        // Count bigrams
        if i < words.len() - 1 {
            let bigram = format!("{} {}", words[i], words[i + 1]);
            if let Some(&idx) = vocab.get(&bigram) {
                vector[idx] += 1;
            }
        }
    }
    vector
}

fn bow_search(query: &str, docs: &[&str], vocab: &HashMap<String, usize>) -> Vec<(usize, usize)> {
    let query_vec = bow_vectorize(query, vocab);
    let mut scores = Vec::new();
    for (i, doc) in docs.iter().enumerate() {
        let doc_vec = bow_vectorize(doc, vocab);
        let score = query_vec.dot(&doc_vec);
        scores.push((i, score));
    }
    scores.sort_by(|a, b| b.1.cmp(&a.1));
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
    println!("Vocabulary size: {}", vocab.len());

    let results = bow_search(query, &knowledge_base, &vocab);
    println!("BOW Search Results:");
    for (idx, score) in results {
        println!(
            "  Doc {idx} | Score: {score} | Text: {}",
            knowledge_base[idx]
        );
    }
}
