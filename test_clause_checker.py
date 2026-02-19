"""
Unit tests for the ClauseChecker class.

These tests verify the functionality of clause detection, matching, and evidence extraction,
including semantic comparison capabilities.
"""

import os
import json
import unittest
from unittest.mock import Mock, patch, MagicMock
from clause_checker import ClauseChecker


class TestClauseChecker(unittest.TestCase):
    """Test cases for the ClauseChecker class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock environment variables
        os.environ['AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT'] = 'https://test.cognitiveservices.azure.com/'
        os.environ['AZURE_DOCUMENT_INTELLIGENCE_KEY'] = 'test-key-12345'

    def tearDown(self):
        """Clean up after tests."""
        # Remove test environment variables
        env_vars_to_cleanup = [
            'AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT',
            'AZURE_DOCUMENT_INTELLIGENCE_KEY',
            'AZURE_OPENAI_ENDPOINT',
            'AZURE_OPENAI_KEY'
        ]
        for var in env_vars_to_cleanup:
            if var in os.environ:
                del os.environ[var]

    @patch('clause_checker.DocumentIntelligenceClient')
    def test_initialization_with_env_vars(self, mock_client):
        """Test that ClauseChecker initializes correctly with environment variables."""
        checker = ClauseChecker()
        self.assertIsNotNone(checker.client)
        self.assertEqual(checker.endpoint, 'https://test.cognitiveservices.azure.com/')
        self.assertEqual(checker.key, 'test-key-12345')

    @patch('clause_checker.DocumentIntelligenceClient')
    def test_initialization_with_params(self, mock_client):
        """Test that ClauseChecker initializes correctly with parameters."""
        endpoint = 'https://custom.cognitiveservices.azure.com/'
        key = 'custom-key'
        checker = ClauseChecker(endpoint=endpoint, key=key)
        self.assertEqual(checker.endpoint, endpoint)
        self.assertEqual(checker.key, key)

    def test_initialization_without_credentials(self):
        """Test that ClauseChecker raises error without credentials."""
        # Clear environment variables
        if 'AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT' in os.environ:
            del os.environ['AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT']
        if 'AZURE_DOCUMENT_INTELLIGENCE_KEY' in os.environ:
            del os.environ['AZURE_DOCUMENT_INTELLIGENCE_KEY']
        
        with self.assertRaises(ValueError):
            ClauseChecker()

    @patch('clause_checker.DocumentIntelligenceClient')
    def test_get_analyzer_config(self, mock_client):
        """Test getting analyzer configuration."""
        checker = ClauseChecker()
        config = checker.get_analyzer_config()
        
        self.assertIsInstance(config, dict)
        self.assertEqual(config['analyzerId'], 'clause-checker')
        self.assertIn('fieldSchema', config)
        self.assertIn('models', config)
        self.assertEqual(config['models']['completion'], 'gpt-4.1-mini')
        self.assertEqual(config['models']['embedding'], 'text-embedding-3-large')

    @patch('clause_checker.DocumentIntelligenceClient')
    def test_update_analyzer_config(self, mock_client):
        """Test updating analyzer configuration."""
        checker = ClauseChecker()
        new_config = {'analyzerId': 'new-analyzer', 'test': 'value'}
        checker.update_analyzer_config(new_config)
        
        self.assertEqual(checker.analyzer_config, new_config)
        self.assertEqual(checker.get_analyzer_config(), new_config)

    @patch('clause_checker.DocumentIntelligenceClient')
    def test_analyze_exact_match(self, mock_client):
        """Test clause analysis with exact match."""
        checker = ClauseChecker()
        
        document_text = "This document contains a confidentiality clause that protects information."
        target_clause = "confidentiality clause"
        
        result = checker._analyze_for_clause(document_text, target_clause)
        
        self.assertTrue(result['clausePresent'])
        self.assertEqual(result['matchType'], 'Exact')
        self.assertEqual(result['confidence'], 1.0)
        self.assertIn('confidentiality clause', result['evidenceQuote'].lower())

    @patch('clause_checker.DocumentIntelligenceClient')
    def test_analyze_paraphrase_match(self, mock_client):
        """Test clause analysis with paraphrase match."""
        checker = ClauseChecker()
        
        # Use a document with clear word overlap
        document_text = "This confidentiality agreement ensures all information remains private and secure at all times."
        target_clause = "confidentiality agreement ensures information private"
        
        result = checker._analyze_for_clause(document_text, target_clause)
        
        # Should find a match (either Semantic or Paraphrase)
        self.assertTrue(result['clausePresent'])
        self.assertIn(result['matchType'], ['Paraphrase', 'Semantic'])
        self.assertGreater(result['confidence'], 0.6)

    @patch('clause_checker.DocumentIntelligenceClient')
    def test_analyze_missing_clause(self, mock_client):
        """Test clause analysis when clause is missing."""
        checker = ClauseChecker()
        
        document_text = "This is a simple document without any special clauses."
        target_clause = "confidentiality agreement"
        
        result = checker._analyze_for_clause(document_text, target_clause)
        
        self.assertFalse(result['clausePresent'])
        self.assertEqual(result['matchType'], 'Missing')
        self.assertEqual(result['confidence'], 0.0)
        self.assertEqual(result['evidenceQuote'], '')

    @patch('clause_checker.DocumentIntelligenceClient')
    def test_find_best_match(self, mock_client):
        """Test finding best matching section."""
        checker = ClauseChecker()
        
        document_text = "First sentence. Second sentence with confidentiality and agreement words. Third sentence."
        target_clause = "confidentiality agreement"
        
        best_match = checker._find_best_match(document_text, target_clause)
        
        self.assertIsInstance(best_match, str)
        self.assertIn('confidentiality', best_match.lower())

    @patch('clause_checker.DocumentIntelligenceClient')
    def test_field_schema_structure(self, mock_client):
        """Test that field schema has correct structure."""
        checker = ClauseChecker()
        config = checker.get_analyzer_config()
        
        fields = config['fieldSchema']['fields']
        
        # Check all required fields exist
        self.assertIn('targetClause', fields)
        self.assertIn('clausePresent', fields)
        self.assertIn('matchType', fields)
        self.assertIn('evidenceQuote', fields)
        
        # Check field types
        self.assertEqual(fields['targetClause']['type'], 'string')
        self.assertEqual(fields['clausePresent']['type'], 'boolean')
        self.assertEqual(fields['matchType']['type'], 'string')
        self.assertEqual(fields['evidenceQuote']['type'], 'string')
        
        # Check methods
        self.assertEqual(fields['targetClause']['method'], 'generate')
        self.assertEqual(fields['clausePresent']['method'], 'classify')
        self.assertEqual(fields['matchType']['method'], 'classify')
        self.assertEqual(fields['evidenceQuote']['method'], 'extract')

    @patch('clause_checker.DocumentIntelligenceClient')
    def test_semantic_similarity_calculation(self, mock_client):
        """Test semantic similarity calculation."""
        checker = ClauseChecker()
        
        # Test similar sentences
        text1 = "The parties agree to maintain confidentiality"
        text2 = "Both sides will keep information private"
        
        similarity = checker._calculate_semantic_similarity(text1, text2)
        
        # Should have some similarity (exact value depends on method used)
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)

    @patch('clause_checker.DocumentIntelligenceClient')
    def test_semantic_comparison_enabled(self, mock_client):
        """Test that semantic comparison can be enabled/disabled."""
        checker = ClauseChecker()
        
        # Check default config has semantic comparison settings
        config = checker.get_analyzer_config()
        self.assertIn('useSemanticComparison', config.get('config', {}))
        self.assertIn('semanticSimilarityThreshold', config.get('config', {}))

    @patch('clause_checker.DocumentIntelligenceClient')
    def test_semantic_match_type(self, mock_client):
        """Test that semantic match type is returned for similar but not exact matches."""
        checker = ClauseChecker()
        
        # Use semantic comparison to find similar clause
        document_text = "The parties must keep all proprietary information confidential and secure."
        target_clause = "confidentiality agreement"
        
        result = checker._analyze_for_clause(document_text, target_clause)
        
        # Result should be deterministic based on the text and semantic comparison
        if result['clausePresent']:
            # If clause is found, it should be via Semantic or Paraphrase match
            self.assertIn(result['matchType'], ['Semantic', 'Paraphrase'])
            self.assertGreater(result['confidence'], 0.0)
        else:
            # If not found, it should be Missing
            self.assertEqual(result['matchType'], 'Missing')
            self.assertEqual(result['confidence'], 0.0)
        
    @patch('clause_checker.DocumentIntelligenceClient')
    def test_find_most_similar_section(self, mock_client):
        """Test finding the most similar section in document."""
        checker = ClauseChecker()
        
        document_text = ("This is an introduction. "
                        "The confidentiality clause requires parties to protect information. "
                        "This is a conclusion.")
        target_clause = "confidentiality agreement"
        
        section, similarity = checker._find_most_similar_section(document_text, target_clause)
        
        self.assertIsInstance(section, str)
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
        # Should contain relevant text
        self.assertIn('confidentiality', section.lower())

    @patch('clause_checker.DocumentIntelligenceClient')
    def test_embedding_models_config(self, mock_client):
        """Test that embedding models are configured."""
        checker = ClauseChecker()
        config = checker.get_analyzer_config()
        
        models = config.get('models', {})
        self.assertIn('embedding', models)
        # Check for fallback model option
        if 'embeddingFallback' in models:
            self.assertIsInstance(models['embeddingFallback'], str)


if __name__ == '__main__':
    unittest.main()
