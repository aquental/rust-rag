// TODO: Write a function to preprocess a string
// Steps:
// - Split the string into words
// - Convert each word to lowercase
// - Remove punctuation from the start and end
// - Return a Vec<String> of cleaned tokens

fn preprocess_string(text: &str) -> Vec<String> {
    text.split_whitespace() // Split into words
        .map(|word| {
            word.to_lowercase() // Convert to lowercase
                .trim_matches(|c: char| c.is_ascii_punctuation()) // Remove punctuation from start/end
                .to_string() // Convert to String
        })
        .collect() // Collect into Vec<String>
}

fn main() {
    let cleaned = preprocess_string("Hello, World! We are preprocessing strings today.");
    println!("{:?}", cleaned);
}
