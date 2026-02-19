"""
Azure Content Understanding Clause Checker

This module provides functionality to check if a specific clause exists in a document
using Azure Document Intelligence (Content Understanding) service.
"""

import os
import json
from typing import Dict, Any, Optional
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv


class ClauseChecker:
    """
    A class to check for clause existence in documents using Azure Content Understanding.
    """

    def __init__(self, endpoint: Optional[str] = None, key: Optional[str] = None):
        """
        Initialize the ClauseChecker with Azure credentials.

        Args:
            endpoint: Azure Document Intelligence endpoint URL
            key: Azure Document Intelligence API key
        """
        # Load environment variables
        load_dotenv()

        self.endpoint = endpoint or os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        self.key = key or os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

        if not self.endpoint or not self.key:
            raise ValueError(
                "Azure credentials not provided. Set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT "
                "and AZURE_DOCUMENT_INTELLIGENCE_KEY environment variables or pass them as arguments."
            )

        # Initialize the client
        self.client = DocumentIntelligenceClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.key)
        )

        # Load analyzer configuration
        self.analyzer_config = self._load_analyzer_config()

    def _load_analyzer_config(self) -> Dict[str, Any]:
        """
        Load the analyzer configuration from the JSON file.

        Returns:
            Dictionary containing the analyzer configuration
        """
        config_path = os.path.join(
            os.path.dirname(__file__),
            "analyzer_config.json"
        )
        
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Return default configuration if file not found
            return {
                "analyzerId": "clause-checker",
                "description": "Checks whether a target clause exists in a document and returns evidence.",
                "baseAnalyzerId": "prebuilt-document",
                "config": {
                    "enableOcr": True,
                    "enableLayout": True,
                    "returnDetails": True,
                    "estimateFieldSourceAndConfidence": True
                },
                "models": {
                    "completion": "gpt-4.1-mini",
                    "embedding": "text-embedding-3-large"
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
                            "estimateSourceAndConfidence": True
                        }
                    }
                }
            }

    def check_clause(self, document_path: str, target_clause: str) -> Dict[str, Any]:
        """
        Check if a target clause exists in the given document.

        Args:
            document_path: Path to the document file to analyze
            target_clause: The clause text to search for

        Returns:
            Dictionary containing:
                - clausePresent (bool): Whether the clause is present
                - matchType (str): "Exact", "Paraphrase", or "Missing"
                - evidenceQuote (str): Supporting quote from the document
                - confidence (float): Confidence score if available
        """
        # Read the document
        with open(document_path, "rb") as f:
            document_bytes = f.read()

        # For now, we'll use the prebuilt-document model to extract content
        # In a production scenario, you would create/use a custom model with the analyzer config
        poller = self.client.begin_analyze_document(
            model_id="prebuilt-document",
            document=document_bytes,
        )
        result = poller.result()

        # Extract all text content from the document
        document_text = ""
        if result.content:
            document_text = result.content

        # Analyze the document for the target clause
        analysis_result = self._analyze_for_clause(document_text, target_clause)

        return analysis_result

    def _analyze_for_clause(self, document_text: str, target_clause: str) -> Dict[str, Any]:
        """
        Analyze document text to determine if the target clause is present.

        Args:
            document_text: Extracted text from the document
            target_clause: The clause to search for

        Returns:
            Dictionary with analysis results
        """
        # Convert to lowercase for comparison
        doc_lower = document_text.lower()
        clause_lower = target_clause.lower()

        # Check for exact match
        if clause_lower in doc_lower:
            # Find the exact location and extract surrounding context
            start_idx = doc_lower.index(clause_lower)
            end_idx = start_idx + len(target_clause)
            
            # Extract evidence with some context
            context_start = max(0, start_idx - 50)
            context_end = min(len(document_text), end_idx + 50)
            evidence = document_text[context_start:context_end].strip()

            return {
                "clausePresent": True,
                "matchType": "Exact",
                "evidenceQuote": evidence,
                "confidence": 1.0,
                "targetClause": target_clause
            }

        # Check for word-by-word match (paraphrase detection - simple version)
        clause_words = set(clause_lower.split())
        doc_words = set(doc_lower.split())
        
        # Calculate word overlap
        overlap = clause_words.intersection(doc_words)
        overlap_ratio = len(overlap) / len(clause_words) if clause_words else 0

        if overlap_ratio > 0.6:  # If more than 60% words match
            # Find best matching section
            evidence = self._find_best_match(document_text, target_clause)
            return {
                "clausePresent": True,
                "matchType": "Paraphrase",
                "evidenceQuote": evidence,
                "confidence": overlap_ratio,
                "targetClause": target_clause
            }

        # No match found
        return {
            "clausePresent": False,
            "matchType": "Missing",
            "evidenceQuote": "",
            "confidence": 0.0,
            "targetClause": target_clause
        }

    def _find_best_match(self, document_text: str, target_clause: str) -> str:
        """
        Find the best matching section in the document for a paraphrase.

        Args:
            document_text: The full document text
            target_clause: The target clause

        Returns:
            Best matching text snippet
        """
        # Split document into sentences or paragraphs
        sentences = document_text.split('.')
        clause_words = set(target_clause.lower().split())
        
        best_score = 0
        best_sentence = ""
        
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(clause_words.intersection(sentence_words))
            score = overlap / len(clause_words) if clause_words else 0
            
            if score > best_score:
                best_score = score
                best_sentence = sentence.strip()
        
        # Return the sentence with some context
        if best_sentence:
            return best_sentence[:200]  # Limit to 200 characters
        return ""

    def get_analyzer_config(self) -> Dict[str, Any]:
        """
        Get the current analyzer configuration.

        Returns:
            Dictionary containing the analyzer configuration
        """
        return self.analyzer_config

    def update_analyzer_config(self, config: Dict[str, Any]) -> None:
        """
        Update the analyzer configuration.

        Args:
            config: New analyzer configuration dictionary
        """
        self.analyzer_config = config
