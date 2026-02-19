"""
Azure AI Content Understanding REST API Client

This module provides REST API functionality to interact with Azure AI Content Understanding
for creating/updating analyzers and analyzing documents.
"""

import os
import json
import requests
import time
from typing import Dict, Any, Optional
from dotenv import load_dotenv


class ContentUnderstandingClient:
    """
    REST API client for Azure AI Content Understanding.
    """
    
    def __init__(self, endpoint: Optional[str] = None, key: Optional[str] = None):
        """
        Initialize the Content Understanding REST client.
        
        Args:
            endpoint: Azure Content Understanding endpoint URL
            key: Azure Content Understanding API key
        """
        load_dotenv()
        
        self.endpoint = endpoint or os.getenv("AZURE_CONTENT_UNDERSTANDING_ENDPOINT")
        self.key = key or os.getenv("AZURE_CONTENT_UNDERSTANDING_KEY")
        
        if not self.endpoint or not self.key:
            raise ValueError(
                "Azure Content Understanding credentials not provided. "
                "Set AZURE_CONTENT_UNDERSTANDING_ENDPOINT and AZURE_CONTENT_UNDERSTANDING_KEY "
                "environment variables or pass them as arguments."
            )
        
        # Remove trailing slash from endpoint if present
        self.endpoint = self.endpoint.rstrip('/')
        
        # API version for Content Understanding
        self.api_version = "2024-11-30"
        
        # Default headers
        self.headers = {
            "Ocp-Apim-Subscription-Key": self.key,
            "Content-Type": "application/json"
        }
    
    def create_or_update_analyzer(
        self, 
        analyzer_id: str, 
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create or update a custom analyzer for Content Understanding.
        
        Args:
            analyzer_id: Unique identifier for the analyzer
            schema: Analyzer schema configuration
            
        Returns:
            Dictionary containing the analyzer creation/update response
            
        Raises:
            requests.HTTPError: If the API request fails
        """
        url = f"{self.endpoint}/documentintelligence/documentModels/{analyzer_id}?api-version={self.api_version}"
        
        # Build the request payload
        payload = {
            "modelId": analyzer_id,
            "description": schema.get("description", "Custom clause checker analyzer"),
            "buildMode": "template",
            "fields": {}
        }
        
        # Add fields from schema
        if "fields" in schema:
            for field in schema["fields"]:
                field_name = field["name"]
                payload["fields"][field_name] = {
                    "type": field.get("type", "string"),
                    "description": field.get("description", "")
                }
        
        # Add Azure OpenAI settings if present
        if "azureOpenAISettings" in schema:
            payload["azureOpenAISettings"] = schema["azureOpenAISettings"]
        
        try:
            response = requests.put(url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            # If the response is 201 Created or 200 OK
            if response.status_code in [200, 201]:
                print(f"✓ Analyzer '{analyzer_id}' created/updated successfully")
                return response.json() if response.text else {"status": "success"}
            else:
                return {"status": "success", "statusCode": response.status_code}
                
        except requests.exceptions.HTTPError as e:
            error_message = f"Failed to create/update analyzer: {e}"
            if e.response is not None:
                try:
                    error_details = e.response.json()
                    error_message += f"\nDetails: {json.dumps(error_details, indent=2)}"
                except Exception:
                    error_message += f"\nResponse: {e.response.text}"
            raise Exception(error_message) from e
    
    def get_analyzer(self, analyzer_id: str) -> Dict[str, Any]:
        """
        Get information about an existing analyzer.
        
        Args:
            analyzer_id: Unique identifier for the analyzer
            
        Returns:
            Dictionary containing analyzer information
        """
        url = f"{self.endpoint}/documentintelligence/documentModels/{analyzer_id}?api-version={self.api_version}"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                raise Exception(f"Analyzer '{analyzer_id}' not found") from e
            raise Exception(f"Failed to get analyzer: {e}") from e
    
    def analyze_document(
        self,
        document_path: str,
        target_clause: str,
        analyzer_id: str = "prebuilt-document"
    ) -> Dict[str, Any]:
        """
        Analyze a document using Content Understanding REST API.
        
        Args:
            document_path: Path to the document file
            target_clause: The clause to search for
            analyzer_id: ID of the analyzer to use (default: prebuilt-document)
            
        Returns:
            Dictionary containing analysis results
        """
        # Read the document
        with open(document_path, "rb") as f:
            document_bytes = f.read()
        
        # Determine content type based on file extension
        content_type = self._get_content_type(document_path)
        
        # Start the analysis
        analyze_url = f"{self.endpoint}/documentintelligence/documentModels/{analyzer_id}:analyze?api-version={self.api_version}"
        
        analyze_headers = {
            "Ocp-Apim-Subscription-Key": self.key,
            "Content-Type": content_type
        }
        
        # Add query parameter for target clause if using custom analyzer
        if analyzer_id != "prebuilt-document":
            analyze_url += f"&targetClause={requests.utils.quote(target_clause)}"
        
        try:
            # Submit document for analysis
            response = requests.post(
                analyze_url,
                headers=analyze_headers,
                data=document_bytes
            )
            response.raise_for_status()
            
            # Get the operation location to poll for results
            operation_location = response.headers.get("Operation-Location")
            if not operation_location:
                # Some APIs return result directly
                return self._parse_analysis_result(response.json(), target_clause)
            
            # Poll for results
            result = self._poll_for_result(operation_location)
            
            # Parse and return the result
            return self._parse_analysis_result(result, target_clause)
            
        except requests.exceptions.HTTPError as e:
            error_message = f"Failed to analyze document: {e}"
            if e.response is not None:
                try:
                    error_details = e.response.json()
                    error_message += f"\nDetails: {json.dumps(error_details, indent=2)}"
                except Exception:
                    error_message += f"\nResponse: {e.response.text}"
            raise Exception(error_message) from e
    
    def _get_content_type(self, file_path: str) -> str:
        """
        Determine content type based on file extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Content-Type header value
        """
        extension = os.path.splitext(file_path)[1].lower()
        
        content_types = {
            ".pdf": "application/pdf",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".tiff": "image/tiff",
            ".tif": "image/tiff",
            ".bmp": "image/bmp"
        }
        
        return content_types.get(extension, "application/octet-stream")
    
    def _poll_for_result(
        self, 
        operation_location: str, 
        max_attempts: int = 60,
        delay: int = 2
    ) -> Dict[str, Any]:
        """
        Poll the operation location until the analysis is complete.
        
        Args:
            operation_location: URL to poll for results
            max_attempts: Maximum number of polling attempts
            delay: Delay between polling attempts in seconds
            
        Returns:
            Analysis result dictionary
        """
        poll_headers = {
            "Ocp-Apim-Subscription-Key": self.key
        }
        
        for attempt in range(max_attempts):
            response = requests.get(operation_location, headers=poll_headers)
            response.raise_for_status()
            
            result = response.json()
            status = result.get("status", "").lower()
            
            if status == "succeeded":
                return result
            elif status == "failed":
                error = result.get("error", {})
                raise Exception(f"Analysis failed: {error.get('message', 'Unknown error')}")
            elif status in ["running", "notstarted"]:
                time.sleep(delay)
            else:
                raise Exception(f"Unknown status: {status}")
        
        raise Exception(f"Analysis timed out after {max_attempts * delay} seconds")
    
    def _parse_analysis_result(
        self, 
        result: Dict[str, Any], 
        target_clause: str
    ) -> Dict[str, Any]:
        """
        Parse the analysis result and extract relevant information.
        
        Args:
            result: Raw analysis result from the API
            target_clause: The target clause being searched for
            
        Returns:
            Parsed result dictionary with clause checking fields
        """
        # Extract content from the analysis result
        content = ""
        
        # Try to get content from different possible locations in the response
        if "analyzeResult" in result:
            analyze_result = result["analyzeResult"]
            content = analyze_result.get("content", "")
            
            # If custom fields are present, use them
            if "documents" in analyze_result and analyze_result["documents"]:
                doc = analyze_result["documents"][0]
                fields = doc.get("fields", {})
                
                # Check if our custom fields are present
                if "clausePresent" in fields or "matchType" in fields:
                    return {
                        "targetClause": target_clause,
                        "clausePresent": fields.get("clausePresent", {}).get("value", False),
                        "matchType": fields.get("matchType", {}).get("value", "Missing"),
                        "evidenceQuote": fields.get("evidenceQuote", {}).get("value", ""),
                        "confidence": fields.get("confidence", {}).get("value", 0.0),
                        "source": fields.get("source", {}).get("value", "")
                    }
        elif "content" in result:
            content = result["content"]
        
        # Fallback: perform basic clause matching on extracted content
        return self._basic_clause_check(content, target_clause)
    
    def _basic_clause_check(
        self, 
        content: str, 
        target_clause: str
    ) -> Dict[str, Any]:
        """
        Perform basic clause checking on extracted content.
        This is a fallback when custom analyzer fields are not available.
        
        Args:
            content: Extracted document content
            target_clause: The target clause to search for
            
        Returns:
            Dictionary with clause checking results
        """
        content_lower = content.lower()
        clause_lower = target_clause.lower()
        
        # Check for exact match
        if clause_lower in content_lower:
            # Find the location
            start_idx = content_lower.index(clause_lower)
            end_idx = start_idx + len(target_clause)
            
            # Extract evidence with context
            context_start = max(0, start_idx - 50)
            context_end = min(len(content), end_idx + 50)
            evidence = content[context_start:context_end].strip()
            
            return {
                "targetClause": target_clause,
                "clausePresent": True,
                "matchType": "Exact",
                "evidenceQuote": evidence,
                "confidence": 1.0,
                "source": ""
            }
        
        # Check for word overlap (simple paraphrase detection)
        clause_words = set(clause_lower.split())
        content_words = set(content_lower.split())
        overlap = clause_words.intersection(content_words)
        overlap_ratio = len(overlap) / len(clause_words) if clause_words else 0.0
        
        if overlap_ratio > 0.6:
            # Find best matching section
            sentences = content.split('.')
            best_score = 0
            best_sentence = ""
            
            for sentence in sentences:
                sentence_words = set(sentence.lower().split())
                sent_overlap = len(clause_words.intersection(sentence_words))
                score = sent_overlap / len(clause_words) if clause_words else 0
                
                if score > best_score:
                    best_score = score
                    best_sentence = sentence.strip()
            
            return {
                "targetClause": target_clause,
                "clausePresent": True,
                "matchType": "Paraphrase",
                "evidenceQuote": best_sentence[:200] if best_sentence else "",
                "confidence": overlap_ratio,
                "source": ""
            }
        
        # No match found
        return {
            "targetClause": target_clause,
            "clausePresent": False,
            "matchType": "Missing",
            "evidenceQuote": "",
            "confidence": 0.0,
            "source": ""
        }
