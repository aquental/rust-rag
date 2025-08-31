use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
struct Document {
    title: String,
    content: String,
}

type KnowledgeBase = HashMap<String, Document>;

fn create_knowledge_base() -> KnowledgeBase {
    let mut kb = HashMap::new();

    kb.insert(
        "AAPL".to_string(),
        Document {
            title: "AAPL Stock (April 2023)".to_string(),
            content: "On 2023-04-13, AAPL opened at $160.50, closed at $162.30, \
                     with a high of $163.00 and a low of $159.90. \
                     Trading volume was 80 million shares. \
                     On 2023-04-14, AAPL opened at $161.10, closed at $162.80, \
                     with a high of $163.50 and a low of $160.50. \
                     Trading volume was 85 million shares.".to_string()
        }
    );

    kb.insert(
        "MSFT".to_string(), 
        Document {
            title: "MSFT Stock (April 2023)".to_string(),
            content: "On 2023-04-13, MSFT opened at $285.00, closed at $288.50, \
                     with a high of $290.00 and a low of $283.50. \
                     Trading volume was 35 million shares. \
                     On 2023-04-14, MSFT opened at $286.00, closed at $289.00, \
                     with a high of $291.50 and a low of $284.70. \
                     Trading volume was 40 million shares.".to_string()
        }
    );

    kb.insert(
        "TSLA".to_string(),
        Document {
            title: "TSLA Stock (April 2023)".to_string(),
            content: "On 2023-04-13, TSLA opened at $185.00, closed at $187.00, \
                     with a high of $189.00 and a low of $184.50. \
                     Trading volume was 50 million shares. \
                     On 2023-04-14, TSLA opened at $186.00, closed at $188.50, \
                     with a high of $190.00 and a low of $185.50. \
                     Trading volume was 55 million shares.".to_string()
        }
    );

    kb
}

/// Retrieve the top K documents from the knowledge base with the highest word overlap.
fn rag_retrieval<'a>(query: &str, documents: &'a KnowledgeBase, k: usize) -> Vec<&'a Document> {
    // Convert the query to lowercase and tokenize
    let query_lower = query.to_lowercase();
    let query_words: HashSet<_> = query_lower
        .split_whitespace()
        .collect();
    
    // Calculate overlap score for each document
    let mut scored_documents: Vec<(&'a Document, usize)> = documents
        .iter()
        .map(|(_, doc)| {
            // Convert document content to lowercase and tokenize
            let content_lower = doc.content.to_lowercase();
            let content_words: HashSet<_> = content_lower
                .split_whitespace()
                .collect();
            
            // Calculate overlap score using set intersection
            let overlap = query_words.intersection(&content_words).count();
            
            (doc, overlap)
        })
        .collect();
    
    // Print the overlap score for each document
    println!("\nDocument relevance scores:");
    for (doc, score) in &scored_documents {
        println!("  {} - Score: {}", doc.title, score);
    }
    println!();
    
    // Sort documents by overlap score in descending order
    scored_documents.sort_by(|a, b| b.1.cmp(&a.1));
    
    // Return the top K documents with the highest overlap scores
    scored_documents
        .into_iter()
        .filter(|(_, score)| *score > 0)  // Only include documents with at least some overlap
        .take(k)
        .map(|(doc, _)| doc)
        .collect()
}

fn main() {
    let kb = create_knowledge_base();

    println!("Testing rag_retrieval function");
    println!("{}", "=".repeat(50));
    
    // Test 1: Query about AAPL
    println!("\nTest 1: Query about AAPL");
    let query1 = "What was AAPL's opening price and volume on April 14?";
    let results1 = rag_retrieval(query1, &kb, 2);
    println!("Query: {}", query1);
    println!("Retrieved {} documents:", results1.len());
    for doc in &results1 {
        println!("  - {}", doc.title);
    }
    
    // Test 2: Query about all stocks
    println!("\nTest 2: Query about all stocks");
    let query2 = "Show me the closing prices for AAPL, MSFT, and TSLA on April 14, 2023";
    let results2 = rag_retrieval(query2, &kb, 3);
    println!("Query: {}", query2);
    println!("Retrieved {} documents:", results2.len());
    for doc in &results2 {
        println!("  - {}", doc.title);
    }
    
    // Test 3: Query with mixed case
    println!("\nTest 3: Query with mixed case (testing case-insensitive matching)");
    let query3 = "TRADING VOLUME shares MILLION april";
    let results3 = rag_retrieval(query3, &kb, 2);
    println!("Query: {}", query3);
    println!("Retrieved {} documents:", results3.len());
    for doc in &results3 {
        println!("  - {}", doc.title);
    }
    
    // Test 4: Query with no matches
    println!("\nTest 4: Query with no matches");
    let query4 = "Bitcoin cryptocurrency blockchain";
    let results4 = rag_retrieval(query4, &kb, 2);
    println!("Query: {}", query4);
    println!("Retrieved {} documents:", results4.len());
    for doc in &results4 {
        println!("  - {}", doc.title);
    }
}
