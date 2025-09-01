use std::collections::{HashMap, HashSet};

fn build_vocab(docs: &[&str]) -> HashMap<String, usize> {
    // Initialize an empty set for unique words
    let mut unique_words = HashSet::new();

    // Iterate through each document and its words
    for doc in docs {
        for word in doc.to_lowercase().split_whitespace() {
            // Clean words by converting to lowercase and removing punctuation
            let clean_word = word.trim_matches(|c: char| ".,!?".contains(c));
            if !clean_word.is_empty() {
                unique_words.insert(clean_word.to_string());
            }
        }
    }

    // Convert set to sorted vector and create word-to-index mapping
    let mut words: Vec<String> = unique_words.into_iter().collect();
    words.sort();

    // Return a dictionary mapping words to indices
    words
        .into_iter()
        .enumerate()
        .map(|(idx, word)| (word, idx))
        .collect()
}

fn bow_vectorize(text: &str, vocab: &HashMap<String, usize>) -> Vec<usize> {
    // Create a zero vector with length equal to vocabulary size
    let mut vector = vec![0; vocab.len()];

    // Process each word in the text
    for word in text.to_lowercase().split_whitespace() {
        // Clean the word by removing punctuation
        let clean_word = word.trim_matches(|c: char| ".,!?".contains(c));

        // If word exists in vocabulary, increment its count in the vector
        if let Some(&index) = vocab.get(clean_word) {
            vector[index] += 1;
        }
    }

    // Return the BOW vector
    vector
}

fn main() {
    let example_texts = vec![
        "RAG stands for retrieval augmented generation, and retrieval is a key component of RAG.",
        "Data is crucial for retrieval processes, and without data, retrieval systems cannot function effectively.",
    ];

    // Build vocabulary from example texts
    let vocab = build_vocab(&example_texts);

    // Print the vocabulary to see word-to-index mapping
    println!("Vocabulary:");
    for (word, idx) in &vocab {
        println!("{}: {}", word, idx);
    }
    println!();

    // Convert each text to its BOW vector and print
    for text in &example_texts {
        let vector = bow_vectorize(text, &vocab);
        println!("Text: {}\nBOW Vector: {:?}\n", text, vector);
    }
}
