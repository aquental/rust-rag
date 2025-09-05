/// Splits the given text into chunks of size 'chunk_size' words.
pub fn chunk_text(text: &str, chunk_size: usize) -> Vec<String> {
    // Split the text into words, filtering out empty strings
    let words: Vec<&str> = text.split_whitespace().filter(|w| !w.is_empty()).collect();

    // Create chunks of up to chunk_size words
    let mut chunks: Vec<String> = Vec::new();

    for i in (0..words.len()).step_by(chunk_size) {
        // Take up to chunk_size words starting from index i
        let chunk: String = words[i..std::cmp::min(i + chunk_size, words.len())].join(" ");
        chunks.push(chunk);
    }

    chunks
}
