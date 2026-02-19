#!/usr/bin/env python3
"""
Update Analyzer Script

This script creates or updates a Content Understanding analyzer from a JSON schema file.
"""

import sys
import json
import argparse
from pathlib import Path
from content_understanding_rest import ContentUnderstandingClient


def load_schema(schema_path: str) -> dict:
    """
    Load analyzer schema from JSON file.
    
    Args:
        schema_path: Path to the schema JSON file
        
    Returns:
        Dictionary containing the schema
    """
    try:
        with open(schema_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Schema file not found: {schema_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in schema file: {e}")
        sys.exit(1)


def main():
    """
    Main function to update analyzer.
    """
    parser = argparse.ArgumentParser(
        description="Create or update an Azure Content Understanding analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update analyzer using default schema
  python update_analyzer.py

  # Update analyzer with custom schema file
  python update_analyzer.py --schema my_schema.json

  # Update analyzer with custom ID
  python update_analyzer.py --analyzer-id my-clause-checker

  # Specify credentials directly
  python update_analyzer.py --endpoint <endpoint> --key <key>

Environment Variables:
  AZURE_CONTENT_UNDERSTANDING_ENDPOINT  Azure Content Understanding endpoint
  AZURE_CONTENT_UNDERSTANDING_KEY       Azure Content Understanding API key
        """
    )
    
    parser.add_argument(
        "--schema",
        default="analyzer_schema.json",
        help="Path to analyzer schema JSON file (default: analyzer_schema.json)"
    )
    
    parser.add_argument(
        "--analyzer-id",
        default="clause-checker",
        help="Analyzer ID (default: clause-checker)"
    )
    
    parser.add_argument(
        "--endpoint",
        help="Azure Content Understanding endpoint (overrides environment variable)"
    )
    
    parser.add_argument(
        "--key",
        help="Azure Content Understanding API key (overrides environment variable)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Azure Content Understanding - Analyzer Update")
    print("=" * 70)
    print()
    
    # Load schema
    print(f"Loading schema from: {args.schema}")
    schema = load_schema(args.schema)
    print(f"✓ Schema loaded successfully")
    print(f"  Description: {schema.get('description', 'N/A')}")
    print(f"  Fields: {len(schema.get('fields', []))}")
    print()
    
    # Initialize client
    try:
        print("Initializing Content Understanding client...")
        client = ContentUnderstandingClient(
            endpoint=args.endpoint,
            key=args.key
        )
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
    
    # Create/update analyzer
    try:
        print(f"Creating/updating analyzer: {args.analyzer_id}")
        print()
        
        result = client.create_or_update_analyzer(
            analyzer_id=args.analyzer_id,
            schema=schema
        )
        
        print()
        print("=" * 70)
        print("✓ SUCCESS")
        print("=" * 70)
        print()
        print(f"Analyzer '{args.analyzer_id}' is ready to use!")
        print()
        print("Next steps:")
        print(f"  1. Test the analyzer with: python check_clause_rest.py <document> <clause>")
        print(f"  2. Or use in your code: from content_understanding_rest import ContentUnderstandingClient")
        print()
        
    except Exception as e:
        print()
        print("=" * 70)
        print("❌ FAILED")
        print("=" * 70)
        print()
        print(f"Error: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Verify your credentials are correct")
        print("  2. Check that your Azure resource supports Content Understanding")
        print("  3. Ensure the schema JSON is valid")
        print("  4. Check Azure service status and region availability")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
