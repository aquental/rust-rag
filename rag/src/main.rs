mod llm;

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

async fn naive_generation(query: &str, llm: &llm::LlmClient) -> Result<String, Box<dyn std::error::Error>> {
    let prompt = format!("Answer directly the following query: {}", query);
    llm.get_llm_response(&prompt).await
}

/// Retrieve all documents with any word overlap with the query.
fn rag_retrieval<'a>(query: &str, documents: &'a KnowledgeBase) -> Vec<&'a Document> {
    let query_lower = query.to_lowercase();
    let query_words: HashSet<_> = query_lower.split_whitespace().map(|s| s.to_string()).collect();

    documents.iter()
        .filter_map(|(_, doc)| {
            let content_lower = doc.content.to_lowercase();
            let content_words: HashSet<_> = content_lower.split_whitespace().map(|s| s.to_string()).collect();
            let overlap = query_words.intersection(&content_words).count();
            if overlap > 0 {
                Some(doc)
            } else {
                None
            }
        })
        .collect()
}

/// Generate a response using the retrieved documents as context.
async fn rag_generation(query: &str, documents: Vec<&Document>, llm: &llm::LlmClient) -> Result<String, Box<dyn std::error::Error>> {
    let prompt = if documents.is_empty() {
        format!("No relevant information found. Answer directly: {}", query)
    } else {
        let context = documents.iter()
            .map(|doc| format!("{}: {}", doc.title, doc.content))
            .collect::<Vec<_>>()
            .join("\n");
        format!("Using the following information:\n'{}'\nAnswer: {}", context, query)
    };
    llm.get_llm_response(&prompt).await
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let kb = create_knowledge_base();
    let query = "What are the applications of Project Chimera?";

    let llm_client = llm::LlmClient::new();

    println!("Naive approach: {}", naive_generation(query, &llm_client).await?);

    let retrieved_docs = rag_retrieval(query, &kb);
    println!("RAG approach: {}", rag_generation(query, retrieved_docs, &llm_client).await?);

    Ok(())
}
