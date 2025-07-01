"""
Judge and Meta-Judge modules for evaluating generation quality and feedback.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
import json

from ..models.llm_interface import LLMInterface, GenerationParams


@dataclass
class JudgmentResult:
    """Result from judge evaluation"""
    alignment_score: float  # 0-1 score for preference alignment
    confidence: float  # 0-1 confidence in the judgment
    feedback_text: str  # Detailed feedback
    suggestions: List[str]  # Specific improvement suggestions
    metadata: Dict[str, Any]  # Additional metadata


class Judge:
    """
    Judge LLM for evaluating alignment between generations and inferred preferences.
    """
    
    def __init__(self, llm: LLMInterface, model_name: str = "gpt-4o-mini"):
        self.llm = llm
        self.model_name = model_name
        self.generation_params = GenerationParams(temperature=0.1)  # Low temperature for consistency
        logging.info(f"Initialized Judge with model: {model_name}")
    
    def evaluate_alignment(self, 
                          prompt: str,
                          generation: str,
                          preferences: Dict[str, Any],
                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate how well a generation aligns with inferred user preferences.
        
        Args:
            prompt: Original prompt
            generation: Generated text to evaluate
            preferences: Inferred user preferences
            context: Additional context information
        
        Returns:
            Dictionary containing judgment results
        """
        judge_prompt = self._create_judge_prompt(prompt, generation, preferences, context)
        
        try:
            response = self.llm.generate(judge_prompt, self.generation_params)
            judgment = self._parse_judge_response(response.text)
            
            # Add metadata
            judgment['judge_prompt'] = judge_prompt
            judgment['raw_response'] = response.text
            judgment['model'] = self.model_name
            
            return judgment
        
        except Exception as e:
            logging.error(f"Error in judge evaluation: {str(e)}")
            return self._create_fallback_judgment()
    
    def _create_judge_prompt(self, 
                           prompt: str,
                           generation: str,
                           preferences: Dict[str, Any],
                           context: Optional[Dict[str, Any]] = None) -> str:
        """Create prompt for judge evaluation"""
        
        # Format preferences
        pref_text = self._format_preferences(preferences)
        
        # Add context if available
        context_text = ""
        if context:
            context_text = f"\n\nAdditional Context:\n{json.dumps(context, indent=2)}"
        
        judge_prompt = f"""You are an expert evaluator assessing how well an AI-generated response aligns with user preferences.

Original Prompt:
{prompt}

Generated Response:
{generation}

Inferred User Preferences:
{pref_text}{context_text}

Please evaluate the generated response based on how well it matches the user's preferences. Consider:
1. Content alignment with stated preferences
2. Style and tone consistency
3. Structural preferences (format, length, etc.)
4. Topic focus and emphasis
5. Overall user satisfaction likelihood

Provide your evaluation in the following JSON format:
{{
    "alignment_score": <float between 0.0 and 1.0>,
    "confidence": <float between 0.0 and 1.0>,
    "feedback_text": "<detailed explanation of your assessment>",
    "suggestions": ["<specific improvement suggestion 1>", "<suggestion 2>", "..."],
    "strengths": ["<what the response did well>", "..."],
    "weaknesses": ["<areas for improvement>", "..."]
}}

Be specific and constructive in your feedback. Justify your alignment score with concrete examples from the text."""
        
        return judge_prompt
    
    def _format_preferences(self, preferences: Dict[str, Any]) -> str:
        """Format preferences for inclusion in prompt"""
        if not preferences:
            return "No specific preferences identified."
        
        formatted = []
        for key, value in preferences.items():
            if isinstance(value, list):
                formatted.append(f"- {key}: {', '.join(str(v) for v in value)}")
            elif isinstance(value, dict):
                formatted.append(f"- {key}: {json.dumps(value, indent=2)}")
            else:
                formatted.append(f"- {key}: {value}")
        
        return "\n".join(formatted)
    
    def _parse_judge_response(self, response_text: str) -> Dict[str, Any]:
        """Parse judge response into structured format"""
        try:
            # Try to extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                parsed = json.loads(json_text)
                
                # Validate required fields
                required_fields = ['alignment_score', 'confidence', 'feedback_text', 'suggestions']
                for field in required_fields:
                    if field not in parsed:
                        raise ValueError(f"Missing required field: {field}")
                
                # Ensure scores are in valid range
                parsed['alignment_score'] = max(0.0, min(1.0, float(parsed['alignment_score'])))
                parsed['confidence'] = max(0.0, min(1.0, float(parsed['confidence'])))
                
                return parsed
            
            else:
                # Fallback parsing if JSON extraction fails
                return self._parse_unstructured_response(response_text)
        
        except Exception as e:
            logging.warning(f"Failed to parse judge response: {str(e)}")
            return self._parse_unstructured_response(response_text)
    
    def _parse_unstructured_response(self, response_text: str) -> Dict[str, Any]:
        """Fallback parsing for unstructured responses"""
        # Simple heuristic parsing
        alignment_score = 0.5  # Default middle score
        confidence = 0.5
        
        # Look for numeric scores in text
        import re
        score_matches = re.findall(r'(\d+\.?\d*)(?:/10|%| out of)', response_text.lower())
        if score_matches:
            try:
                score = float(score_matches[0])
                if score > 1:  # Assume it's out of 10 or percentage
                    score = score / 10 if score <= 10 else score / 100
                alignment_score = max(0.0, min(1.0, score))
            except ValueError:
                pass
        
        return {
            'alignment_score': alignment_score,
            'confidence': confidence,
            'feedback_text': response_text,
            'suggestions': ["Could not parse specific suggestions"],
            'strengths': [],
            'weaknesses': [],
            'parsing_error': True
        }
    
    def _create_fallback_judgment(self) -> Dict[str, Any]:
        """Create fallback judgment when evaluation fails"""
        return {
            'alignment_score': 0.5,
            'confidence': 0.0,
            'feedback_text': "Unable to evaluate due to system error",
            'suggestions': ["System error occurred during evaluation"],
            'strengths': [],
            'weaknesses': [],
            'error': True
        }


class MetaJudge:
    """
    Meta-Judge for evaluating judge quality and providing feedback on judgments.
    """
    
    def __init__(self, llm: LLMInterface, model_name: str = "gpt-4o-mini"):
        self.llm = llm
        self.model_name = model_name
        self.generation_params = GenerationParams(temperature=0.1)
        logging.info(f"Initialized MetaJudge with model: {model_name}")
    
    def evaluate_judge(self, 
                      original_prompt: str,
                      generation: str,
                      judge_feedback: Dict[str, Any],
                      preferences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the quality of judge feedback and suggest improvements.
        
        Args:
            original_prompt: Original prompt
            generation: Generated text that was judged
            judge_feedback: Judge's evaluation
            preferences: User preferences used in judgment
        
        Returns:
            Meta-judge evaluation of the judge
        """
        meta_prompt = self._create_meta_judge_prompt(
            original_prompt, generation, judge_feedback, preferences
        )
        
        try:
            response = self.llm.generate(meta_prompt, self.generation_params)
            meta_judgment = self._parse_meta_judge_response(response.text)
            
            # Add metadata
            meta_judgment['meta_judge_prompt'] = meta_prompt
            meta_judgment['raw_response'] = response.text
            meta_judgment['model'] = self.model_name
            
            return meta_judgment
        
        except Exception as e:
            logging.error(f"Error in meta-judge evaluation: {str(e)}")
            return self._create_fallback_meta_judgment()
    
    def _create_meta_judge_prompt(self, 
                                original_prompt: str,
                                generation: str,
                                judge_feedback: Dict[str, Any],
                                preferences: Dict[str, Any]) -> str:
        """Create prompt for meta-judge evaluation"""
        
        judge_feedback_text = json.dumps(judge_feedback, indent=2)
        preferences_text = json.dumps(preferences, indent=2)
        
        meta_prompt = f"""You are a meta-evaluator assessing the quality of an AI judge's evaluation. Your job is to critique the judge's assessment and suggest improvements.

Original Prompt:
{original_prompt}

Generated Response:
{generation}

User Preferences:
{preferences_text}

Judge's Evaluation:
{judge_feedback_text}

Please evaluate the judge's assessment considering:
1. Accuracy: Does the alignment score accurately reflect preference matching?
2. Specificity: Are the suggestions concrete and actionable?
3. Completeness: Did the judge consider all relevant preference aspects?
4. Consistency: Is the feedback logically consistent with the score?
5. Usefulness: Would this feedback help improve future generations?

Provide your meta-evaluation in JSON format:
{{
    "judge_quality_score": <float 0.0-1.0>,
    "accuracy_assessment": "<evaluation of score accuracy>",
    "feedback_quality": "<assessment of feedback usefulness>",
    "missing_aspects": ["<what the judge missed>", "..."],
    "judge_strengths": ["<what the judge did well>", "..."],
    "revised_suggestions": ["<improved suggestions for the original generation>", "..."],
    "meta_confidence": <float 0.0-1.0>
}}

Be constructive and specific in your critique."""
        
        return meta_prompt
    
    def _parse_meta_judge_response(self, response_text: str) -> Dict[str, Any]:
        """Parse meta-judge response"""
        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                parsed = json.loads(json_text)
                
                # Validate and normalize scores
                if 'judge_quality_score' in parsed:
                    parsed['judge_quality_score'] = max(0.0, min(1.0, float(parsed['judge_quality_score'])))
                if 'meta_confidence' in parsed:
                    parsed['meta_confidence'] = max(0.0, min(1.0, float(parsed['meta_confidence'])))
                
                return parsed
            
            else:
                return {
                    'judge_quality_score': 0.5,
                    'accuracy_assessment': response_text,
                    'feedback_quality': "Could not parse structured response",
                    'missing_aspects': [],
                    'judge_strengths': [],
                    'revised_suggestions': [],
                    'meta_confidence': 0.3,
                    'parsing_error': True
                }
        
        except Exception as e:
            logging.warning(f"Failed to parse meta-judge response: {str(e)}")
            return self._create_fallback_meta_judgment()
    
    def _create_fallback_meta_judgment(self) -> Dict[str, Any]:
        """Create fallback meta-judgment when evaluation fails"""
        return {
            'judge_quality_score': 0.5,
            'accuracy_assessment': "Unable to assess due to system error",
            'feedback_quality': "System error occurred",
            'missing_aspects': [],
            'judge_strengths': [],
            'revised_suggestions': [],
            'meta_confidence': 0.0,
            'error': True
        } 