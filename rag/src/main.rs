mod llm;

use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
struct Document {
    title: String,
    content: String,
}

type KnowledgeBase = HashMap<String, Document>;

/// Creates a `KnowledgeBase` containing three sample documents related to Project Chimera.
/// These documents are used for testing and demonstration purposes.
///
/// The `KnowledgeBase` is a `HashMap` where the keys are document IDs and the values
/// are `Document` structs containing the title and content of each document.
///
/// Document 1: "Project Chimera Overview"
/// Document 2: "Chimera's Neural Interface"
/// Document 3: "Applications of Chimera"
fn create_knowledge_base() -> KnowledgeBase {
    let mut kb = HashMap::new();

    kb.insert(
        "doc1".to_string(),
        Document {
            title: "Project Chimera Overview".to_string(),
            content: "Project Chimera is a research initiative focused on developing \
                      novel bio-integrated interfaces. It aims to merge biological \
                      systems with advanced computing technologies."
                .to_string(),
        },
    );

    kb.insert(
        "doc2".to_string(),
        Document {
            title: "Chimera's Neural Interface".to_string(),
            content: "The core component of Project Chimera is a neural interface \
                      that allows for bidirectional communication between the brain \
                      and external devices. This interface uses biocompatible \
                      nanomaterials."
                .to_string(),
        },
    );

    kb.insert(
        "doc3".to_string(),
        Document {
            title: "Applications of Chimera".to_string(),
            content: "Potential applications of Project Chimera include advanced \
                      prosthetics, treatment of neurological disorders, and enhanced \
                      human-computer interaction. Ethical considerations are paramount."
                .to_string(),
        },
    );

    kb
}

/// naive_generation function
/// Parameters:
///   query: &str - The user's question
///   llm: &llm::LlmClient - The LLM client instance
/// Returns: Result<String, Box<dyn std::error::Error>>
/// Steps:
///   1. Format a simple prompt with the query using format!()
///   2. Call llm.get_llm_response() with the prompt and return result
async fn naive_generation(
    query: &str,
    llm: &llm::LlmClient,
) -> Result<String, Box<dyn std::error::Error>> {
    let prompt = format!("Answer directly the following query: {}", query);
    llm.get_llm_response(&prompt).await
}

/// rag_retrieval function
/// Parameters:
///   query: &str - The user's question
///   documents: &'a KnowledgeBase - The knowledge base
/// Returns: Vec<&'a Document> - A vector of documents that overlap with the query
/// Steps:
///   1. Convert query to lowercase and collect words into HashSet
///   2. Iterate through documents using iter() and filter_map()
///   3. For each doc, get word overlap count with query
///   4. Return documents with overlap count > 0 using collect()
fn rag_retrieval<'a>(query: &str, documents: &'a KnowledgeBase) -> Vec<&'a Document> {
    let query_lower = query.to_lowercase();
    let query_words: HashSet<_> = query_lower
        .split_whitespace()
        .map(|s| s.to_string())
        .collect();

    documents
        .iter()
        .filter_map(|(_, doc)| {
            let content_lower = doc.content.to_lowercase();
            let content_words: HashSet<_> = content_lower
                .split_whitespace()
                .map(|s| s.to_string())
                .collect();
            let overlap = query_words.intersection(&content_words).count();
            if overlap > 0 { Some(doc) } else { None }
        })
        .collect()
}

/// rag_generation function
/// Parameters:
///   query: &str - The user's question
///   documents: Vec<&Document> - The retrieved documents
///   llm: &llm::LlmClient - The LLM client instance
/// Returns: Result<String, Box<dyn std::error::Error>>
/// Steps:
///   1. Match on documents to create appropriate prompt
///   2. If documents is empty, create direct query prompt
///   3. If documents is not empty, create prompt with context from documents
///   4. Call llm.get_llm_response() with prompt and return result
async fn rag_generation(
    query: &str,
    documents: Vec<&Document>,
    llm: &llm::LlmClient,
) -> Result<String, Box<dyn std::error::Error>> {
    let prompt = if documents.is_empty() {
        format!("No relevant information found. Answer directly: {}", query)
    } else {
        let context = documents
            .iter()
            .map(|doc| format!("{}: {}", doc.title, doc.content))
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "Using the following information:\n'{}'\nAnswer: {}",
            context, query
        )
    };
    llm.get_llm_response(&prompt).await
}

/// Main entry point for the RAG application.
///
/// This function initializes a knowledge base of documents, asks a user for a query,
/// and then uses both a naive and a RAG-based approach to generate an answer.
/// The two answers are then printed to the console for comparison.
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let kb = create_knowledge_base();
    let query = "What are the applications of Project Chimera?";

    // Create LlmClient instance
    let llm_client = llm::LlmClient::new();

    // Call naive_generation and print result
    println!(
        "Naive approach: {}",
        naive_generation(query, &llm_client).await?
    );

    // Call rag_retrieval to get relevant document
    let retrieved_docs = rag_retrieval(query, &kb);
    // Call rag_generation with document and print result
    println!(
        "RAG approach: {}",
        rag_generation(query, retrieved_docs, &llm_client).await?
    );

    Ok(())
}
