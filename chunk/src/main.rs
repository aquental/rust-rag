mod data;

fn main() {
    // Example text to test the chunking
    let sample_text = "This is a sample text that we will use to test our chunking function. It contains multiple sentences to make it more interesting.";

    // Call the chunk_text function with the sample text and chunk size of 5
    let chunks = data::chunk_text(sample_text, 5);

    // Print each chunk on a new line to see how the text was split
    for (i, chunk) in chunks.iter().enumerate() {
        println!("Chunk {}: {}", i + 1, chunk);
    }
}
