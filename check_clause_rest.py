#!/usr/bin/env python3
"""
Check Clause Script (REST API)

This script analyzes a document for clause existence using Azure Content Understanding REST API.
"""

import sys
import json
import argparse
from pathlib import Path
from content_understanding_rest import ContentUnderstandingClient


def main():
    """
    Main function to check clause in document.
    """
    parser = argparse.ArgumentParser(
        description="Check if a clause exists in a document using Azure Content Understanding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python check_clause_rest.py contract.pdf "confidentiality clause"

  # Use custom analyzer
  python check_clause_rest.py contract.pdf "payment terms" --analyzer-id my-analyzer

  # Save results to file
  python check_clause_rest.py contract.pdf "liability clause" --output results.json

  # Specify credentials directly
  python check_clause_rest.py contract.pdf "clause" --endpoint <endpoint> --key <key>

Environment Variables:
  AZURE_CONTENT_UNDERSTANDING_ENDPOINT  Azure Content Understanding endpoint
  AZURE_CONTENT_UNDERSTANDING_KEY       Azure Content Understanding API key
        """
    )
    
    parser.add_argument(
        "document",
        help="Path to the document file to analyze (PDF, DOCX, images, etc.)"
    )
    
    parser.add_argument(
        "clause",
        help="The target clause to search for in the document"
    )
    
    parser.add_argument(
        "--analyzer-id",
        default="prebuilt-document",
        help="Analyzer ID to use (default: prebuilt-document)"
    )
    
    parser.add_argument(
        "--endpoint",
        help="Azure Content Understanding endpoint (overrides environment variable)"
    )
    
    parser.add_argument(
        "--key",
        help="Azure Content Understanding API key (overrides environment variable)"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        help="Output file to save results (JSON format)"
    )
    
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Quiet mode - only output results"
    )
    
    args = parser.parse_args()
    
    # Check if document exists
    if not Path(args.document).exists():
        print(f"❌ Error: Document not found: {args.document}")
        sys.exit(1)
    
    if not args.quiet:
        print("=" * 70)
        print("Azure Content Understanding - Clause Checker (REST API)")
        print("=" * 70)
        print()
        print(f"Document: {args.document}")
        print(f"Target Clause: {args.clause}")
        print(f"Analyzer: {args.analyzer_id}")
        print()
    
    # Initialize client
    try:
        if not args.quiet:
            print("Initializing Content Understanding client...")
        
        client = ContentUnderstandingClient(
            endpoint=args.endpoint,
            key=args.key
        )
        
        if not args.quiet:
            print(f"✓ Client initialized")
            print(f"  Endpoint: {client.endpoint}")
            print()
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print()
        print("Please set environment variables or provide credentials:")
        print("  export AZURE_CONTENT_UNDERSTANDING_ENDPOINT=<your-endpoint>")
        print("  export AZURE_CONTENT_UNDERSTANDING_KEY=<your-key>")
        print()
        print("Or use --endpoint and --key arguments")
        sys.exit(1)
    
    # Analyze document
    try:
        if not args.quiet:
            print("Analyzing document...")
            print("(This may take a few moments)")
            print()
        
        result = client.analyze_document(
            document_path=args.document,
            target_clause=args.clause,
            analyzer_id=args.analyzer_id
        )
        
        # Display results
        if not args.quiet:
            print("=" * 70)
            print("ANALYSIS RESULTS")
            print("=" * 70)
            print()
        
        print(f"Target Clause: {result['targetClause']}")
        print(f"Clause Present: {'✓ Yes' if result['clausePresent'] else '✗ No'}")
        print(f"Match Type: {result['matchType']}")
        print(f"Confidence: {result['confidence']:.2%}")
        
        if result.get('source'):
            print(f"Source: {result['source']}")
        
        if result.get('evidenceQuote'):
            print()
            print("Evidence Quote:")
            print("-" * 70)
            print(result['evidenceQuote'])
            print("-" * 70)
        
        print()
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            
            if not args.quiet:
                print(f"✓ Results saved to: {args.output}")
                print()
        
        if not args.quiet:
            print("=" * 70)
            print("✓ Analysis complete")
            print("=" * 70)
        
        # Exit with appropriate code
        sys.exit(0 if result['clausePresent'] else 1)
        
    except FileNotFoundError as e:
        print(f"❌ File Error: {e}")
        sys.exit(1)
    except Exception as e:
        print()
        print("=" * 70)
        print("❌ Analysis Failed")
        print("=" * 70)
        print()
        print(f"Error: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Verify your credentials are correct")
        print("  2. Check that the document file is valid and not corrupted")
        print("  3. Ensure the analyzer ID exists (or use 'prebuilt-document')")
        print("  4. Check Azure service status and API limits")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
