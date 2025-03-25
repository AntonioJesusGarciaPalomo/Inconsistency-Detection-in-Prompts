"""Service for interacting with the Azure OpenAI API with enhanced capabilities."""
import logging
import asyncio
from typing import Dict, Any, List, Optional
import json
import re
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from app.core.config import settings

logger = logging.getLogger(__name__)

class OpenAIService:
    """Service for interfacing with Azure OpenAI."""
    
    def __init__(self):
        """Initialize the OpenAI service with Azure API configuration."""
        self.api_key = settings.openai_api_key
        self.endpoint = settings.openai_endpoint.rstrip("/") if settings.openai_endpoint else ""
        self.api_version = settings.openai_api_version
        self.deployment = settings.openai_deployment
        
        if not self.api_key or not self.endpoint or not self.deployment:
            logger.warning("Azure OpenAI credentials not fully configured")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_completion(
        self, 
        system_message: str, 
        user_message: str, 
        temperature: float = 0.0,
        max_tokens: int = 2000
    ) -> str:
        """Get a completion from the Azure OpenAI API."""
        if not self.api_key or not self.endpoint:
            raise ValueError("Azure OpenAI credentials not configured")
        
        url = f"{self.endpoint}/openai/deployments/{self.deployment}/chat/completions?api-version={self.api_version}"
        
        payload = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            
            if response.status_code != 200:
                logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
                raise ValueError(f"OpenAI API error: {response.status_code} - {response.text}")
            
            data = response.json()
            return data["choices"][0]["message"]["content"]
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def batch_evaluate_consistency(
        self, 
        claim_pairs: List[List[str]], 
        temperature: float = 0.0
    ) -> List[float]:
        """Evaluate the consistency of multiple pairs of claims in parallel."""
        if not claim_pairs:
            return []
        
        system_prompt = """
        You are an expert at detecting logical inconsistencies between statements.
        Analyze the logical relationship between the two statements and rate their consistency 
        on a scale of 0-10, where:
        - 0 means completely contradictory (the statements logically cannot both be true)
        - 5 means unrelated or neutral (no logical connection)
        - 10 means completely consistent (the statements logically support or complement each other)
        
        Important instructions:
        1. Focus ONLY on logical consistency, not factual truth
        2. Pay special attention to comparative relationships (greater than, older than, etc.)
        3. Look for transitive relationship contradictions (A > B > C > A patterns)
        4. Return ONLY the numerical score as a single number (do not include any explanatory text)
        """
        
        tasks = []
        for pair in claim_pairs:
            if len(pair) != 2:
                logger.warning(f"Invalid claim pair format: {pair}")
                continue
                
            user_message = f"Statement 1: {pair[0]}\nStatement 2: {pair[1]}\n\nConsistency score (0-10):"
            tasks.append(self.get_completion(system_prompt, user_message, temperature))
        
        if not tasks:
            return []
            
        try:
            results = await asyncio.gather(*tasks)
            
            # Convert string results to float scores
            scores = []
            for result in results:
                try:
                    # Extract just the number from the response
                    match = re.search(r'\b(\d+(?:\.\d+)?)\b', result)
                    if match:
                        score = float(match.group(1))
                        # Ensure the score is in the valid range
                        score = max(0, min(10, score))
                        scores.append(score)
                    else:
                        logger.warning(f"Could not extract number from: '{result}'")
                        scores.append(5.0)  # Default to neutral
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not parse consistency score from '{result}': {e}")
                    scores.append(5.0)  # Default to neutral
                    
            return scores
            
        except Exception as e:
            logger.error(f"Error in batch evaluation: {str(e)}")
            # Return neutral scores as fallback
            return [5.0] * len(tasks)

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=5))
    async def analyze_logical_consistency(self, text: str) -> Dict[str, Any]:
        """
        Analyze a text for logical consistency in a single call.
        This is a more efficient approach than multiple separate calls.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary with analysis results
        """
        system_message = """
        Analyze the following text for logical consistency. Follow these steps:
        
        1. Identify individual claims in the text
        2. Detect any logical inconsistencies or contradictions between claims
        3. Look for circular relationships or transitive inconsistencies
        
        Return your analysis as a valid JSON object with these fields:
        {
          "claims": [list of individual claims extracted from the text],
          "inconsistencies_detected": true/false,
          "inconsistency_description": "Description of any inconsistencies found",
          "inconsistent_claim_indices": [[0,1,2], [3,4]], // Arrays of claim indices that form inconsistent cycles
          "consistency_score": 0-10 // Overall consistency score (0 = inconsistent, 10 = consistent)
        }
        
        IMPORTANT: Make sure to return only a valid JSON object without any additional text, markdown formatting, or code block markers.
        """
        
        try:
            response = await self.get_completion(
                system_message=system_message,
                user_message=text,
                max_tokens=2500
            )
            
            try:
                # Clean up the response to handle Markdown formatting
                cleaned_response = response
                
                # Remove Markdown JSON code blocks if present
                if "```json" in cleaned_response:
                    cleaned_response = cleaned_response.replace("```json", "").replace("```", "")
                elif "```" in cleaned_response:
                    cleaned_response = cleaned_response.replace("```", "")
                
                # Fix common JSON formatting issues
                # Remove any trailing commas before closing brackets
                cleaned_response = re.sub(r',\s*}', '}', cleaned_response)
                cleaned_response = re.sub(r',\s*]', ']', cleaned_response)
                
                # Remove duplicate keys (keeping the last one)
                # This is a simplistic approach - for production, consider a more robust solution
                parsed_json = json.loads(cleaned_response)
                return parsed_json
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {response}")
                logger.error(f"JSON parse error: {str(e)}")
                
                # Try one more approach - use regex to extract key parts
                try:
                    # Extract consistency score
                    score_match = re.search(r'"consistency_score":\s*(\d+(?:\.\d+)?)', response)
                    consistency_score = float(score_match.group(1)) if score_match else 5.0
                    
                    # Extract inconsistency detected
                    inconsistency_match = re.search(r'"inconsistencies_detected":\s*(true|false)', response, re.IGNORECASE)
                    inconsistencies_detected = inconsistency_match.group(1).lower() == "true" if inconsistency_match else False
                    
                    return {
                        "claims": [],
                        "inconsistencies_detected": inconsistencies_detected,
                        "consistency_score": consistency_score,
                        "inconsistency_description": "Extracted from malformed JSON"
                    }
                except Exception:
                    # Last resort fallback
                    return {
                        "claims": [],
                        "inconsistencies_detected": False,
                        "consistency_score": 5.0,
                        "error": "Failed to parse analysis"
                    }
                
        except Exception as e:
            logger.error(f"Error in logical consistency analysis: {str(e)}")
            return {
                "claims": [],
                "inconsistencies_detected": False,
                "consistency_score": 5.0,
                "error": str(e)
            }