# Retrieval using embeddings

We'll delve into retrieving the most relevant chunks for any given query and building an LLM prompt to produce more accurate, context-driven answers.

## Retrieving the Most Relevant Chunks

Before your LLM can generate a coherent, context-rich answer, we need to fetch the right information. The vector database (Chroma) will rank which document chunks are most relevant for a given query.

# setup

```shell
chroma run --path ./chroma &
```

---

# Category Filtering in RAG System

## Overview

The RAG (Retrieval-Augmented Generation) system now supports category-based filtering to retrieve more relevant documents based on predefined categories.

## Available Categories

Based on the corpus.json data, the following categories are available:

- `"Technology"` - Technology-related content
- `"Science"` - Scientific discoveries and research
- `"Health"` - Healthcare and medical topics
- `"Education"` - Educational content
- `"Business"` - Business and finance topics
- `"Transportation"` - Transportation-related content
- `"Environment"` - Environmental topics
- `"Sports"` - Sports-related content
- `"Entertainment"` - Entertainment content
- `"Travel"` - Travel and tourism
- `"History"` - Historical content
- `"Art & Design"` - Art and design topics
- `"Culture"` - Cultural topics
- `"Psychology"` - Psychology-related content
- `"Literature"` - Literary content
- `"Lifestyle"` - Lifestyle topics

## Usage Examples

### 1. Filter by Technology Category

```rust
let user_query = "What are the latest developments in artificial intelligence?";
let category_filter = Some("Technology");
```

### 2. Filter by Science Category

```rust
let user_query = "What are the recent discoveries in space exploration?";
let category_filter = Some("Science");
```

### 3. Filter by Health Category

```rust
let user_query = "What are the innovations in healthcare?";
let category_filter = Some("Health");
```

### 4. No Category Filter (Search All)

```rust
let user_query = "Tell me about recent innovations";
let category_filter = None;
```

## How It Works

1. **Query with Filter**: When a category is specified, the system uses ChromaDB's metadata filtering to retrieve only documents from that category.

2. **Fallback Mechanism**: If no documents are found with the specified category filter, the system automatically tries again without the filter and notifies the user.

3. **Context Building**: The filtered chunks are passed to the LLM with proper context formatting, ensuring the AI response is based only on relevant documents.

4. **Metadata Storage**: Each document chunk stores its category in ChromaDB metadata, enabling efficient filtering during retrieval.

## Implementation Details

### Vector Database (src/vector_db.rs)

```rust
pub async fn retrieve_top_chunks(
    collection: &ChromaCollection,
    query: &str,
    top_k: usize,
    embedder: &SentenceEmbedder,
    category_filter: Option<&str>,  // New parameter for category filtering
) -> Result<Vec<RetrievedChunk>, Box<dyn std::error::Error>>
```

The function builds a metadata filter when a category is provided:

```rust
let where_metadata = category_filter.map(|category| {
    json!({"category": category})
});
```

### Main Application (src/main.rs)

The main function demonstrates:

- Setting up query and category filter
- Retrieving filtered chunks
- Handling empty results with fallback
- Building prompts with filtered context
- Getting LLM responses based on filtered documents

## Testing

To test different categories, modify the `category_filter` variable in `src/main.rs`:

```bash
# Run with current settings
cargo run

# Output will show:
# - Query being used
# - Category filter applied (if any)
# - Retrieved documents from that category
# - Formatted prompt with context
# - LLM response based on filtered context
```

## Benefits

1. **Improved Relevance**: Responses are based on documents from the specific domain of interest.
2. **Reduced Noise**: Filtering eliminates irrelevant documents from other categories.
3. **Flexible Querying**: Option to search with or without category constraints.
4. **Graceful Fallback**: Automatic retry without filter if category yields no results.
5. **Better Context**: LLM receives more focused context for generating responses.
