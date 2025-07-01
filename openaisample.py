"""
OpenAI API utilities for AI Agent Sandbox Prototype
Handles API calls, error handling, and fallback mechanisms.
"""

import json
import time
from typing import Dict, Any, Optional, Union
from openai import OpenAI
from config import Config


class OpenAIClient:
    """OpenAI API client with error handling and fallback mechanisms"""
    
    def __init__(self):
        """Initialize OpenAI client"""
        self.client = None
        self.is_configured = False
        
        if Config.OPENAI_API_KEY:
            try:
                self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
                self.is_configured = True
            except Exception as e:
                print(f"❌ Failed to initialize OpenAI client: {e}")
                self.is_configured = False
    
    def call_openai_api(self, messages: list, system_prompt: Optional[str] = None, 
                       max_retries: Optional[int] = None) -> Optional[str]:
        """
        Make OpenAI API call with error handling and retries
        
        Args:
            messages (list): List of messages for the conversation
            system_prompt (str): System prompt for the agent
            max_retries (int): Maximum number of retries
            
        Returns:
            Optional[str]: Response from OpenAI API or None if failed
        """
        
        if not self.is_configured:
            return None
        
        max_retries = max_retries or Config.MAX_RETRIES
        
        # Prepare messages with system prompt
        formatted_messages = []
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})
        formatted_messages.extend(messages)
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=Config.OPENAI_MODEL,
                    messages=formatted_messages,
                    max_tokens=Config.OPENAI_MAX_TOKENS,
                    temperature=Config.OPENAI_TEMPERATURE,
                    timeout=Config.TIMEOUT_SECONDS
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                print(f"⚠️  OpenAI API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    # Wait before retrying (exponential backoff)
                    wait_time = 2 ** attempt
                    print(f"   Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print("❌ OpenAI API call failed after all retries")
                    return None
        
        return None
    
    def get_structured_response(self, messages: list, system_prompt: str, 
                              expected_fields: list) -> Optional[Dict[str, Any]]:
        """
        Get structured JSON response from OpenAI API
        
        Args:
            messages (list): List of messages
            system_prompt (str): System prompt
            expected_fields (list): Expected fields in the response
            
        Returns:
            Optional[Dict[str, Any]]: Parsed JSON response or None
        """
        
        # Add instruction for JSON format
        json_instruction = f"\n\nPlease respond with a JSON object that includes these fields: {', '.join(expected_fields)}"
        enhanced_system_prompt = system_prompt + json_instruction
        
        response = self.call_openai_api(messages, enhanced_system_prompt)
        
        if not response:
            return None
        
        try:
            # Try to extract JSON from the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                parsed_response = json.loads(json_str)
                return parsed_response
            else:
                # If no JSON found, create a simple response
                return {"response": response.strip()}
                
        except json.JSONDecodeError as e:
            print(f"⚠️  Failed to parse JSON from OpenAI response: {e}")
            return {"response": response.strip()}
    
    def test_connection(self) -> bool:
        """
        Test OpenAI API connection
        
        Returns:
            bool: True if connection is successful
        """
        
        if not self.is_configured:
            return False
        
        try:
            test_messages = [{"role": "user", "content": "Test message"}]
            response = self.call_openai_api(test_messages, max_retries=1)
            return response is not None
            
        except Exception as e:
            print(f"❌ OpenAI connection test failed: {e}")
            return False


# Global OpenAI client instance
openai_client = OpenAIClient()


def get_ai_response(messages: list, system_prompt: str, 
                   expected_fields: list = None) -> Union[str, Dict[str, Any], None]:
    """
    Get AI response using OpenAI API
    
    Args:
        messages (list): List of messages
        system_prompt (str): System prompt for the agent
        expected_fields (list): Expected fields for structured response
        
    Returns:
        Union[str, Dict[str, Any], None]: AI response or None if failed
    """
    
    if not openai_client.is_configured:
        return None
    
    if expected_fields:
        return openai_client.get_structured_response(messages, system_prompt, expected_fields)
    else:
        return openai_client.call_openai_api(messages, system_prompt)


def check_openai_status() -> Dict[str, Any]:
    """
    Check OpenAI API status and configuration
    
    Returns:
        Dict[str, Any]: Status information
    """
    
    status = {
        'configured': openai_client.is_configured,
        'api_key_set': Config.OPENAI_API_KEY is not None,
        'model': Config.OPENAI_MODEL,
        'connection_test': False
    }
    
    if openai_client.is_configured:
        status['connection_test'] = openai_client.test_connection()
    
    return status 