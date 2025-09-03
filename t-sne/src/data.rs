pub fn get_sentences_and_categories() -> (Vec<String>, Vec<String>) {
    let sentences = vec![
        // Existing sentences
        "Natural language processing involves analyzing text data.".to_string(),
        "Tokenization is a key step in preparing text for models.".to_string(),
        "Transformers have revolutionized NLP tasks.".to_string(),
        "Machine learning models require large datasets.".to_string(),
        "Deep learning excels in complex pattern recognition.".to_string(),
        "Overfitting is a common issue in training ML models.".to_string(),
        "Pizza is a popular dish with many regional variations.".to_string(),
        "Sushi requires precise preparation techniques.".to_string(),
        "Baking bread involves kneading and proofing dough.".to_string(),
        "Rainy days can affect outdoor activities.".to_string(),
        "Temperature fluctuations impact crop growth.".to_string(),
        "Hurricanes are powerful tropical storms.".to_string(),
        // New Travel sentences
        "Exploring ancient ruins offers a glimpse into history.".to_string(),
        "Backpacking through Europe is a popular adventure.".to_string(),
        "Tropical beaches attract tourists year-round.".to_string(),
        "Cultural festivals provide unique travel experiences.".to_string(),
    ];

    let categories = vec![
        // Existing categories
        "NLP".to_string(),
        "NLP".to_string(),
        "NLP".to_string(),
        "ML".to_string(),
        "ML".to_string(),
        "ML".to_string(),
        "Food".to_string(),
        "Food".to_string(),
        "Food".to_string(),
        "Weather".to_string(),
        "Weather".to_string(),
        "Weather".to_string(),
        // New Travel categories
        "Travel".to_string(),
        "Travel".to_string(),
        "Travel".to_string(),
        "Travel".to_string(),
    ];

    (sentences, categories)
}
