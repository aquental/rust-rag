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

# Dual Filtering in RAG System

## Overview

The enhanced RAG system now supports **dual filtering** - combining both category-based filtering and similarity distance thresholds to retrieve the most relevant documents. This provides fine-grained control over the retrieval process.

## Key Features

### 1. Distance Threshold Filtering

- **Purpose**: Ensures only sufficiently similar documents are retrieved
- **Range**: 0.0 (identical) to 2.0 (completely different)
- **Recommended Values**:
  - `0.0 - 0.5`: Very high similarity (nearly identical content)
  - `0.5 - 0.8`: High similarity (closely related content)
  - `0.8 - 1.2`: Good similarity (related content)
  - `1.2 - 1.5`: Moderate similarity (somewhat related)
  - `> 1.5`: Low similarity (loosely related)

### 2. Category Filtering

- Filters documents by predefined categories
- Available categories: Technology, Science, Health, Education, Business, etc.
- Can be combined with distance threshold for precise filtering

### 3. Dual Filtering

- Both filters work simultaneously
- Documents must meet BOTH criteria to be included
- Provides maximum precision in retrieval

## Implementation Details

### Function Signature

```rust
pub async fn retrieve_top_chunks(
    collection: &ChromaCollection,
    query: &str,
    top_k: usize,
    embedder: &SentenceEmbedder,
    category_filter: Option<&str>,     // Category constraint
    distance_threshold: Option<f32>,   // Similarity constraint
) -> Result<Vec<RetrievedChunk>, Box<dyn std::error::Error>>
```

### Filtering Logic

1. **Category Filter**: Applied via ChromaDB's metadata filtering
2. **Distance Filter**: Applied post-retrieval on similarity scores
3. **Optimization**: Requests 3x the desired results when distance filtering is active to ensure enough results after filtering

## Usage Examples

### Example 1: Strict Dual Filtering

```rust
let category_filter = Some("Technology");
let distance_threshold = Some(0.8);  // High similarity required
// Returns only Technology documents with distance ≤ 0.8
```

### Example 2: Category Only

```rust
let category_filter = Some("Health");
let distance_threshold = None;
// Returns all Health documents regardless of similarity
```

### Example 3: Distance Only

```rust
let category_filter = None;
let distance_threshold = Some(1.0);
// Returns documents from any category with good similarity
```

### Example 4: No Filtering

```rust
let category_filter = None;
let distance_threshold = None;
// Returns top-k most similar documents from any category
```

## Graceful Error Handling

### When No Results Match

The system provides helpful feedback when filters are too restrictive:

1. **Clear Error Message**: Indicates which filters were applied
2. **Suggestions**: Offers ways to relax constraints
3. **Fallback Search**: Automatically attempts search without filters to show what's available
4. **Preview Results**: Shows distance scores and previews of unfiltered results

### Example Output

```
⚠️  No relevant documents found!

The search returned no results that meet your criteria:
  • Category: Technology
  • Similarity threshold: distance ≤ 0.50

Suggestions:
  1. Try relaxing the distance threshold (increase the value)
  2. Remove or change the category filter
  3. Rephrase your query

------------------------------------------------------------
Attempting search without filters for comparison...

Found 3 documents without filters:
  1. Distance: 0.8631, Doc ID: 1
     Preview: Artificial intelligence is transforming...
```

## Visual Similarity Indicators

The system provides visual feedback for similarity levels:

- `★★★★★` - Very High (distance ≤ 0.5)
- `★★★★` - High (distance ≤ 0.8)
- `★★★` - Good (distance ≤ 1.0)
- `★★` - Moderate (distance ≤ 1.2)
- `★` - Low (distance > 1.2)

## Performance Considerations

1. **Query Optimization**: When distance filtering is active, the system requests more initial results (3x top_k) to ensure enough results after filtering
2. **Early Termination**: Stops processing once top_k results are collected
3. **Metadata Indexing**: Category filtering leverages ChromaDB's indexed metadata for efficient filtering

## Testing the Feature

### Run with Current Settings

```bash
cargo run
```

### Modify Filters

Edit these lines in `src/main.rs`:

```rust
let category_filter = Some("Technology");  // or None
let distance_threshold = Some(1.0);        // or None
```

### Test Scenarios

See `test_dual_filtering.rs` for comprehensive test scenarios covering:

- Strict dual filtering
- Single filter modes
- No filtering
- Edge cases

## Benefits

1. **Precision**: Retrieve only the most relevant documents
2. **Flexibility**: Use one, both, or neither filter as needed
3. **Performance**: Avoid processing irrelevant documents
4. **User Experience**: Clear feedback when filters are too restrictive
5. **Quality Control**: Ensure LLM receives high-quality context

## Best Practices

1. **Start Broad**: Begin with relaxed thresholds and tighten as needed
2. **Monitor Results**: Check the distance scores to calibrate thresholds
3. **Category Selection**: Choose categories that align with your query domain
4. **Fallback Strategy**: Always handle the case where no results match
5. **Testing**: Test with various combinations to understand the corpus distribution
