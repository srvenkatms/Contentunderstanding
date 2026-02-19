"""
Example usage of the Azure Content Understanding Clause Checker.

This script demonstrates how to use the ClauseChecker class to analyze documents
for the presence of specific clauses.
"""

import sys
import json
from clause_checker import ClauseChecker


def main():
    """
    Main function to demonstrate clause checking functionality.
    """
    print("=" * 60)
    print("Azure Content Understanding - Clause Checker")
    print("=" * 60)
    print()

    # Example usage with environment variables
    try:
        # Initialize the clause checker
        print("Initializing Azure Content Understanding client...")
        checker = ClauseChecker()
        print("✓ Client initialized successfully")
        print()

        # Display analyzer configuration
        config = checker.get_analyzer_config()
        print("Analyzer Configuration:")
        print("-" * 60)
        print(f"Analyzer ID: {config['analyzerId']}")
        print(f"Description: {config['description']}")
        print(f"Base Analyzer: {config['baseAnalyzerId']}")
        print()
        print("Models:")
        print(f"  - Completion: {config['models']['completion']}")
        print(f"  - Embedding: {config['models']['embedding']}")
        print()
        print("Field Schema:")
        for field_name, field_config in config['fieldSchema']['fields'].items():
            print(f"  - {field_name}:")
            print(f"      Type: {field_config['type']}")
            print(f"      Method: {field_config['method']}")
            print(f"      Description: {field_config['description']}")
        print()
        print("=" * 60)
        print()

        # Check if document path and clause are provided as arguments
        if len(sys.argv) < 3:
            print("Usage: python example_usage.py <document_path> <target_clause>")
            print()
            print("Example:")
            print('  python example_usage.py contract.pdf "confidentiality clause"')
            print()
            print("Note: Make sure to set the following environment variables:")
            print("  - AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
            print("  - AZURE_DOCUMENT_INTELLIGENCE_KEY")
            print()
            print("Or copy .env.example to .env and fill in your credentials.")
            return

        document_path = sys.argv[1]
        target_clause = sys.argv[2]

        print(f"Analyzing document: {document_path}")
        print(f"Looking for clause: {target_clause}")
        print()
        print("Processing...")
        print()

        # Check for the clause
        result = checker.check_clause(document_path, target_clause)

        # Display results
        print("=" * 60)
        print("ANALYSIS RESULTS")
        print("=" * 60)
        print()
        print(f"Target Clause: {result['targetClause']}")
        print(f"Clause Present: {'✓ Yes' if result['clausePresent'] else '✗ No'}")
        print(f"Match Type: {result['matchType']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print()
        
        if result['evidenceQuote']:
            print("Evidence Quote:")
            print("-" * 60)
            print(result['evidenceQuote'])
            print("-" * 60)
        print()

        # Export results to JSON
        output_file = "clause_check_result.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to: {output_file}")
        print()

    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print()
        print("Please ensure you have set up your Azure credentials:")
        print("1. Copy .env.example to .env")
        print("2. Fill in your Azure Document Intelligence endpoint and key")
        print()
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"❌ File Error: {e}")
        print()
        print("Please ensure the document file exists and the path is correct.")
        print()
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        print()
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
