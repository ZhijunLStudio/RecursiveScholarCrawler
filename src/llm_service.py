# src/llm_service.py - LLM API interface with metrics tracking

import json
import re
import logging
import time
import datetime
from openai import OpenAI
from src.config import DEFAULT_PROMPT
from src.utils import get_timestamp_str

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, api_key, api_base, model, temperature=None, top_p=None, 
                 top_k=None, presence_penalty=None, max_tokens=None):
        """Initialize LLM service with API parameters."""
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.max_tokens = max_tokens
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )
    
    def extract_paper_info(self, text):
        """Query LLM to extract paper information with usage tracking."""
        # Prepare API parameters
        params = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": DEFAULT_PROMPT.format(paper_text=text[:10000] + ("..." if len(text) > 10000 else ""))}
            ],
        }
        
        # Add optional parameters if specified
        if self.max_tokens:
            params["max_tokens"] = self.max_tokens
        if self.temperature:
            params["temperature"] = self.temperature
        if self.top_p:
            params["top_p"] = self.top_p
        if self.presence_penalty:
            params["presence_penalty"] = self.presence_penalty
        
        # Add extra_body parameters if top_k is specified
        if self.top_k:
            params["extra_body"] = {
                "top_k": self.top_k,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        
        # Track usage data
        usage_data = {
            "timestamp": get_timestamp_str(),
            "model": self.model,
            "input_text_length": len(text),
            "input_text_preview": text[:500] + ("..." if len(text) > 500 else ""),
            "prompt_template": DEFAULT_PROMPT.replace("{paper_text}", "[PAPER_TEXT]")
        }
        
        # Query LLM with timing
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(**params)
            end_time = time.time()
            
            # Record response data
            response_text = response.choices[0].message.content
            usage_data["api_call_seconds"] = round(end_time - start_time, 2)
            usage_data["completion_tokens"] = response.usage.completion_tokens if hasattr(response, 'usage') else None
            usage_data["prompt_tokens"] = response.usage.prompt_tokens if hasattr(response, 'usage') else None
            usage_data["total_tokens"] = response.usage.total_tokens if hasattr(response, 'usage') else None
            
            # Save both full response and preview
            usage_data["response_full"] = response_text  # Full response
            usage_data["response_preview"] = response_text[:500] + ("..." if len(response_text) > 500 else "")
            usage_data["success"] = True
            
            # Parse JSON response
            result = self._parse_json_response(response_text)
            if result:
                # Add LLM usage data to result
                result["llm_usage"] = usage_data
                return result
            else:
                usage_data["error"] = "Failed to parse JSON response"
                return {"llm_usage": usage_data}
                
        except Exception as e:
            end_time = time.time()
            logger.error(f"Error querying LLM: {e}")
            usage_data["api_call_seconds"] = round(end_time - start_time, 2)
            usage_data["error"] = str(e)
            usage_data["success"] = False
            return {"llm_usage": usage_data}
    
    def _parse_json_response(self, response_text):
        """Parse JSON from LLM response with enhanced error handling."""
        try:
            # Clean the text first
            clean_text = response_text.strip()
            
            # Try direct parsing
            return json.loads(clean_text)
        except json.JSONDecodeError:
            # Try extracting JSON from code blocks
            json_pattern = re.compile(r'```(?:json)?\s*(.*?)\s*```', re.DOTALL)
            match = json_pattern.search(response_text)
            if match:
                try:
                    return json.loads(match.group(1))
                except:
                    pass
            
            # Try finding JSON object boundaries
            if '{' in clean_text and '}' in clean_text:
                start_idx = clean_text.find('{')
                end_idx = clean_text.rfind('}') + 1
                try:
                    return json.loads(clean_text[start_idx:end_idx])
                except:
                    pass
            
            # All attempts failed
            logger.error(f"Failed to parse LLM response as JSON: {response_text[:300]}...")
            return {}
