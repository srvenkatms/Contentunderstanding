"""
Azure Content Understanding Clause Checker

This module provides functionality to check if a specific clause exists in a document
using Azure Document Intelligence (Content Understanding) service with semantic comparison.
"""

import os
import json
from typing import Dict, Any, Optional, List
import numpy as np
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

try:
    from openai import AzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


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
        
        # Initialize Azure OpenAI client if available and credentials provided
        self.openai_client = None
        self.openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.openai_key = os.getenv("AZURE_OPENAI_KEY")
        
        if OPENAI_AVAILABLE and self.openai_endpoint and self.openai_key:
            try:
                self.openai_client = AzureOpenAI(
                    azure_endpoint=self.openai_endpoint,
                    api_key=self.openai_key,
                    api_version="2024-02-01"
                )
            except Exception as e:
                print(f"Warning: Could not initialize Azure OpenAI client: {e}")
                self.openai_client = None

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
                    "estimateFieldSourceAndConfidence": True,
                    "useSemanticComparison": True,
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

    def _get_embedding_openai(self, text: str, model: str = "text-embedding-3-small") -> Optional[List[float]]:
        """
        Get text embedding using Azure OpenAI.

        Args:
            text: Text to embed
            model: Embedding model to use

        Returns:
            Embedding vector or None if failed
        """
        if not self.openai_client:
            return None
        
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model=model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Warning: OpenAI embedding failed: {e}")
            return None

    def _get_embedding_tfidf(self, texts: List[str]) -> np.ndarray:
        """
        Get TF-IDF based embeddings as fallback.

        Args:
            texts: List of texts to embed

        Returns:
            TF-IDF vectors
        """
        if not SKLEARN_AVAILABLE:
            return None
        
        try:
            vectorizer = TfidfVectorizer(max_features=384)
            return vectorizer.fit_transform(texts).toarray()
        except Exception as e:
            print(f"Warning: TF-IDF embedding failed: {e}")
            return None

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        # Try Azure OpenAI embeddings first
        if self.openai_client and SKLEARN_AVAILABLE:
            embedding_model = self.analyzer_config.get("models", {}).get("embeddingFallback", "text-embedding-3-small")
            emb1 = self._get_embedding_openai(text1, embedding_model)
            emb2 = self._get_embedding_openai(text2, embedding_model)
            
            if emb1 and emb2:
                # Calculate cosine similarity
                emb1_array = np.array(emb1).reshape(1, -1)
                emb2_array = np.array(emb2).reshape(1, -1)
                similarity = cosine_similarity(emb1_array, emb2_array)[0][0]
                return float(similarity)
        
        # Fallback to TF-IDF based similarity
        if SKLEARN_AVAILABLE:
            embeddings = self._get_embedding_tfidf([text1, text2])
            if embeddings is not None and len(embeddings) == 2:
                similarity = cosine_similarity(embeddings[0:1], embeddings[1:2])[0][0]
                return float(similarity)
        
        # Ultimate fallback: word overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0.0

    def _find_most_similar_section(self, document_text: str, target_clause: str, 
                                   window_size: int = 200) -> tuple[str, float]:
        """
        Find the most semantically similar section in the document.

        Args:
            document_text: Full document text
            target_clause: Target clause to match
            window_size: Size of sliding window in characters

        Returns:
            Tuple of (best matching text, similarity score)
        """
        if len(document_text) < window_size:
            similarity = self._calculate_semantic_similarity(document_text, target_clause)
            return document_text, similarity
        
        best_similarity = 0.0
        best_section = ""
        
        # Split into sentences for better context
        sentences = [s.strip() for s in document_text.split('.') if s.strip()]
        
        # Check each sentence and surrounding context
        for i, sentence in enumerate(sentences):
            # Build context window (current sentence + neighbors)
            start_idx = max(0, i - 1)
            end_idx = min(len(sentences), i + 2)
            context = '. '.join(sentences[start_idx:end_idx])
            
            similarity = self._calculate_semantic_similarity(context, target_clause)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_section = context
        
        return best_section, best_similarity

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

        # Check if semantic comparison is enabled
        use_semantic = self.analyzer_config.get("config", {}).get("useSemanticComparison", True)
        semantic_threshold = self.analyzer_config.get("config", {}).get("semanticSimilarityThreshold", 0.75)
        
        if use_semantic:
            # Use semantic similarity to find matches
            best_section, similarity = self._find_most_similar_section(document_text, target_clause)
            
            if similarity >= semantic_threshold:
                return {
                    "clausePresent": True,
                    "matchType": "Semantic",
                    "evidenceQuote": best_section[:200] if best_section else "",
                    "confidence": float(similarity),
                    "targetClause": target_clause
                }
            elif similarity > 0.6:
                # Lower confidence semantic match
                return {
                    "clausePresent": True,
                    "matchType": "Paraphrase",
                    "evidenceQuote": best_section[:200] if best_section else "",
                    "confidence": float(similarity),
                    "targetClause": target_clause
                }
        
        # Fallback to word-overlap method for paraphrase detection
        # (used when semantic is disabled OR when semantic similarity is too low)
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
