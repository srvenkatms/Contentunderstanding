# Azure Content Understanding - Clause Checker

A Python application that uses Azure Document Intelligence (Content Understanding) to check whether a specific clause exists in a document and returns evidence.

📖 **[Read the Complete Technical Documentation](./TECHNICAL_DOCUMENTATION.md)** - Comprehensive guide covering Azure Content Understanding, LLMs integration, and Python architecture.

## Features

- 🔍 **Clause Detection**: Automatically detects if a target clause exists in documents
- 📄 **Multiple Formats**: Supports PDF, DOCX, images, and other document formats
- 🎯 **Match Types**: Identifies Exact matches, Semantic matches, Paraphrases, or Missing clauses
- 💡 **Evidence Extraction**: Provides quotes from the document that support the decision
- 🤖 **AI-Powered**: Uses GPT-4.1-mini for completion and text-embedding models for semantic comparison
- 🧠 **Semantic Comparison**: Advanced paraphrase detection using embeddings with fallback to TF-IDF for lower environments
- 📊 **Confidence Scores**: Returns confidence levels for each analysis

## Custom Analyzer Configuration

The application uses a custom analyzer with the following configuration:

```json
{
  "analyzerId": "clause-checker",
  "description": "Checks whether a target clause exists in a document and returns evidence.",
  "baseAnalyzerId": "prebuilt-document",
  "config": {
    "enableOcr": true,
    "enableLayout": true,
    "returnDetails": true,
    "estimateFieldSourceAndConfidence": true,
    "useSemanticComparison": true,
    "semanticSimilarityThreshold": 0.75
  },
  "models": {
    "completion": "gpt-4.1-mini",
    "embedding": "text-embedding-3-large",
    "embeddingFallback": "text-embedding-3-small"
  },
  "fieldSchema": {
    "name": "ClauseCheck",
    "fields": {
      "targetClause": {
        "type": "string",
        "description": "The clause we are looking for (provided by the caller).",
        "method": "generate"
      },
      "clausePresent": {
        "type": "boolean",
        "description": "True if the document contains the clause or a clear paraphrase; otherwise false.",
        "method": "classify"
      },
      "matchType": {
        "type": "string",
        "description": "Return one of: Exact, Paraphrase, Missing.",
        "method": "classify"
      },
      "evidenceQuote": {
        "type": "string",
        "description": "A short quote from the document that supports the decision.",
        "method": "extract",
        "estimateSourceAndConfidence": true
      }
    }
  }
}
```

## Prerequisites

- Python 3.8 or higher
- Azure subscription with Document Intelligence (formerly Form Recognizer) resource
- Azure Document Intelligence API key and endpoint
- (Optional) Azure OpenAI resource for enhanced semantic comparison

## Installation

1. Clone the repository:
```bash
git clone https://github.com/srvenkatms/Contentunderstanding.git
cd Contentunderstanding
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Azure credentials:
```bash
cp .env.example .env
```

4. Edit `.env` file with your Azure credentials:
```
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=your-api-key-here

# Optional: For enhanced semantic comparison with Azure OpenAI embeddings
AZURE_OPENAI_ENDPOINT=https://your-openai-resource.openai.azure.com/
AZURE_OPENAI_KEY=your-openai-api-key-here
```

## Usage

### Basic Usage

```python
from clause_checker import ClauseChecker

# Initialize the checker
checker = ClauseChecker()

# Check for a clause in a document
result = checker.check_clause(
    document_path="contract.pdf",
    target_clause="confidentiality agreement"
)

# Access results
print(f"Clause Present: {result['clausePresent']}")
print(f"Match Type: {result['matchType']}")
print(f"Evidence: {result['evidenceQuote']}")
print(f"Confidence: {result['confidence']}")
```

### Command Line Usage

```bash
python example_usage.py <document_path> <target_clause>
```

Example:
```bash
python example_usage.py contract.pdf "confidentiality clause"
```

### Result Format

The analysis returns a dictionary with the following fields:

```python
{
    "targetClause": "confidentiality clause",
    "clausePresent": True,
    "matchType": "Exact",  # or "Semantic", "Paraphrase", or "Missing"
    "evidenceQuote": "...text from document...",
    "confidence": 1.0
}
```

## API Reference

### ClauseChecker Class

#### `__init__(endpoint: Optional[str] = None, key: Optional[str] = None)`
Initialize the ClauseChecker with Azure credentials.

**Parameters:**
- `endpoint` (str, optional): Azure Document Intelligence endpoint URL
- `key` (str, optional): Azure Document Intelligence API key

If not provided, credentials are loaded from environment variables.

#### `check_clause(document_path: str, target_clause: str) -> Dict[str, Any]`
Check if a target clause exists in the given document.

**Parameters:**
- `document_path` (str): Path to the document file to analyze
- `target_clause` (str): The clause text to search for

**Returns:**
- Dictionary containing analysis results with fields:
  - `clausePresent` (bool): Whether the clause is present
  - `matchType` (str): "Exact", "Paraphrase", or "Missing"
  - `evidenceQuote` (str): Supporting quote from the document
  - `confidence` (float): Confidence score
  - `targetClause` (str): The original target clause

#### `get_analyzer_config() -> Dict[str, Any]`
Get the current analyzer configuration.

#### `update_analyzer_config(config: Dict[str, Any]) -> None`
Update the analyzer configuration.

## Documentation

- **[README.md](./Readme.md)** - Quick start guide and basic usage
- **[TECHNICAL_DOCUMENTATION.md](./TECHNICAL_DOCUMENTATION.md)** - Complete technical documentation covering:
  - Azure Content Understanding role and integration
  - LLM integration (GPT-4 and embeddings)
  - Python code architecture and design patterns
  - System components and data flow
  - Semantic comparison implementation
  - API reference with code examples
  - Configuration management
  - Performance optimization
  - Security best practices
- **[QUICKSTART.md](./QUICKSTART.md)** - Quick setup and usage examples

## Project Structure

```
Contentunderstanding/
├── .env.example                    # Example environment variables
├── .gitignore                      # Git ignore rules
├── analyzer_config.json            # Custom analyzer configuration
├── clause_checker.py               # Main ClauseChecker class
├── example_usage.py                # Example usage script
├── requirements.txt                # Python dependencies
├── test_clause_checker.py          # Unit tests
├── Readme.md                       # This file
├── QUICKSTART.md                   # Quick start guide
└── TECHNICAL_DOCUMENTATION.md      # Complete technical documentation
```

## How It Works

1. **Document Loading**: The application reads the document using Azure Document Intelligence
2. **Content Extraction**: Extracts text content using OCR and layout analysis
3. **Clause Analysis**: Analyzes the text to find:
   - **Exact matches**: Direct text matching
   - **Semantic matches**: Similar meaning using AI embeddings (when Azure OpenAI is configured)
   - **Paraphrases**: Similar clauses detected via TF-IDF similarity or word overlap
   - **Missing**: Clause not found in the document
4. **Evidence Collection**: Extracts relevant quotes from the document
5. **Result Generation**: Returns structured results with confidence scores

## Semantic Comparison

The application supports multiple levels of semantic comparison to work in different environments:

### Tier 1: Azure OpenAI Embeddings (Recommended)
- Uses Azure OpenAI text embedding models for high-quality semantic similarity
- Best accuracy for detecting paraphrased clauses
- Requires Azure OpenAI resource and API key

### Tier 2: TF-IDF + Cosine Similarity (Fallback)
- Uses scikit-learn's TF-IDF vectorization for semantic comparison
- Works without Azure OpenAI credentials
- Good accuracy for most use cases
- Built-in fallback for lower environments

### Tier 3: Word Overlap (Ultimate Fallback)
- Simple word-by-word matching using Jaccard similarity
- No external dependencies beyond Python standard library
- Basic but reliable for exact word matches

The system automatically chooses the best available method, making it suitable for any environment from development to production.

## Match Types

- **Exact**: The target clause appears verbatim in the document (confidence: 1.0)
- **Semantic**: High semantic similarity using embeddings (confidence: ≥0.75)
- **Paraphrase**: Similar clause with different wording (confidence: 0.6-0.75)
- **Missing**: The clause was not found in the document (confidence: 0.0)

## Error Handling

The application handles various error scenarios:
- Missing or invalid Azure credentials
- Document file not found
- Unsupported document formats
- API connection issues

## Limitations

- Large documents may take longer to process
- Some document formats may require specific preprocessing

## Configuration

You can customize the semantic comparison behavior by updating the `analyzer_config.json`:

```json
{
  "config": {
    "useSemanticComparison": true,  // Enable/disable semantic comparison
    "semanticSimilarityThreshold": 0.75  // Minimum score for semantic matches (0.0-1.0)
  }
}
```

## Future Enhancements

- [ ] Support for batch document processing
- [ ] Custom confidence thresholds
- [ ] Support for multiple clause checking in a single call
- [ ] Web API interface
- [ ] Document preprocessing and caching

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is provided as-is for demonstration purposes.

## Support

For issues related to:
- **Azure Document Intelligence**: Check [Azure Documentation](https://learn.microsoft.com/azure/ai-services/document-intelligence/)
- **This Application**: Open an issue on GitHub

## Authors

- srvenkatms

## Acknowledgments

- Azure Document Intelligence team for the robust document analysis capabilities
- Azure OpenAI for advanced AI models
