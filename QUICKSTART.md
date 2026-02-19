# Quick Start Guide

## Setup (2 minutes)

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Azure credentials**:
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

3. **Verify installation**:
   ```bash
   python -m unittest test_clause_checker
   ```

## Usage Examples

### Example 1: Check for a confidentiality clause
```bash
python example_usage.py contract.pdf "confidentiality agreement"
```

### Example 2: Programmatic usage
```python
from clause_checker import ClauseChecker

# Initialize
checker = ClauseChecker()

# Analyze document
result = checker.check_clause("document.pdf", "termination clause")

# Check results
if result['clausePresent']:
    print(f"Found: {result['matchType']} match")
    print(f"Evidence: {result['evidenceQuote']}")
else:
    print("Clause not found")
```

### Example 3: Using with custom config
```python
checker = ClauseChecker(
    endpoint="https://your-resource.cognitiveservices.azure.com/",
    key="your-key-here"
)

# Get current config
config = checker.get_analyzer_config()
print(f"Analyzer: {config['analyzerId']}")

# Check clause
result = checker.check_clause("legal_doc.pdf", "liability limitation")
```

## Expected Output Format

```json
{
  "targetClause": "confidentiality clause",
  "clausePresent": true,
  "matchType": "Exact",  // or "Semantic", "Paraphrase", "Missing"
  "evidenceQuote": "...relevant text from document...",
  "confidence": 1.0
}
```

## Match Types

| Type | Description | Confidence |
|------|-------------|------------|
| **Exact** | Clause appears verbatim | 1.0 |
| **Semantic** | High semantic similarity using embeddings | ≥ 0.75 |
| **Paraphrase** | Similar meaning, different wording | 0.6 - 0.75 |
| **Missing** | Clause not found | 0.0 |

## Semantic Comparison

The application automatically uses the best available semantic comparison method:
- **Azure OpenAI Embeddings** (if configured) - Best accuracy
- **TF-IDF Similarity** (fallback) - Good accuracy, no external API needed
- **Word Overlap** (ultimate fallback) - Basic matching

To use Azure OpenAI embeddings, add these to your `.env`:
```
AZURE_OPENAI_ENDPOINT=https://your-openai-resource.openai.azure.com/
AZURE_OPENAI_KEY=your-openai-api-key-here
```

## Troubleshooting

### Issue: "Azure credentials not provided"
**Solution**: Set environment variables or pass credentials to constructor

### Issue: "File not found"
**Solution**: Check document path is correct and file exists

### Issue: "Module not found"
**Solution**: Run `pip install -r requirements.txt`

## Next Steps

1. Test with your own documents
2. Adjust confidence thresholds if needed
3. Customize analyzer configuration for specific use cases
4. Integrate into your application workflow
