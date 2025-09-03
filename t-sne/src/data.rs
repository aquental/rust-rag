pub fn get_sentences_and_categories() -> (Vec<String>, Vec<String>) {
    let sentences = vec![
        // NLP
        "RAG stands for Retrieval-Augmented Generation.",
        "Retrieval is a crucial aspect of modern NLP systems.",
        "Generating text with correct facts is challenging.",
        "Large language models can generate coherent text.",
        "GPT models have billions of parameters.",
        "Natural Language Processing enables computers to understand human language.",
        "Word embeddings capture semantic relationships between words.",
        "Transformer architectures revolutionized NLP research.",
        // ML
        "Machine learning benefits from large datasets.",
        "Supervised learning requires labeled data.",
        "Reinforcement learning is inspired by behavioral psychology.",
        "Neural networks can learn complex functions.",
        "Overfitting is a common problem in ML.",
        "Unsupervised learning uncovers hidden patterns in data.",
        "Feature engineering is critical for model performance.",
        "Cross-validation helps in assessing model generalization.",
        // Food
        "Bananas are commonly used in smoothies.",
        "Oranges are rich in vitamin C.",
        "Pizza is a popular Italian dish.",
        "Cooking pasta requires boiling water.",
        "Chocolate can be sweet or bitter.",
        "Fresh salads are a healthy and refreshing meal.",
        "Sushi combines rice, fish, and seaweed in a delicate balance.",
        "Spices can transform simple ingredients into gourmet dishes.",
        // Weather
        "It often rains in the Amazon rainforest.",
        "Summers can be very hot in the desert.",
        "Hurricanes form over warm ocean waters.",
        "Snowstorms can disrupt transportation.",
        "A sunny day can lift people's mood.",
        "Foggy mornings are common in coastal regions.",
        "Winter brings frosty nights and chilly winds.",
        "Thunderstorms can produce lightning and heavy rain.",
    ]
    .into_iter()
    .map(|s| s.to_string())
    .collect();

    let mut categories = Vec::new();
    categories.extend(vec!["NLP".to_string(); 8]);
    categories.extend(vec!["ML".to_string(); 8]);
    categories.extend(vec!["Food".to_string(); 8]);
    categories.extend(vec!["Weather".to_string(); 8]);

    (sentences, categories)
}
