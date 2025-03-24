"""Service for interacting with the Azure OpenAI API."""
import os
import logging
import asyncio
from typing import Dict, Any, List, Optional
import json
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
        Rate the consistency between the two statements on a scale of 0-10, where:
        - 0 means completely contradictory
        - 10 means perfectly consistent
        
        Return only the numerical score.
        """
        
        tasks = []
        for pair in claim_pairs:
            if len(pair) != 2:
                logger.warning(f"Invalid claim pair format: {pair}")
                continue
                
            user_message = f"Statement 1: {pair[0]}\nStatement 2: {pair[1]}\nConsistency score (0-10):"
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
                    number_str = ''.join(c for c in result if c.isdigit() or c == '.')
                    score = float(number_str)
                    # Ensure the score is in the valid range
                    score = max(0, min(10, score))
                    scores.append(score)
                except ValueError:
                    logger.warning(f"Could not parse consistency score from '{result}'")
                    scores.append(5.0)  # Default to neutral
                    
            return scores
            
        except Exception as e:
            logger.error(f"Error in batch evaluation: {str(e)}")
            # Return neutral scores as fallback
            return [5.0] * len(tasks)