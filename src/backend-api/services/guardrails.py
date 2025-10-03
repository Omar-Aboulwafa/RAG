# services/guardrails.py - Comprehensive Security & Compliance Guardrails

import re
import logging
from typing import Dict, List, Any, Tuple, Optional
from enum import Enum
import json

logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

class GuardrailViolation(Exception):
    def __init__(self, message: str, threat_level: ThreatLevel, violation_type: str):
        self.message = message
        self.threat_level = threat_level
        self.violation_type = violation_type
        super().__init__(message)

class InputGuardrails:
    """Input validation and prompt injection protection"""
    
    # Enhanced prompt injection patterns
    PROMPT_INJECTION_PATTERNS = [
        # Direct instruction overrides
        (r'ignore\s+(?:all\s+)?previous\s+instructions?', ThreatLevel.CRITICAL),
        (r'forget\s+(?:everything|all\s+previous)', ThreatLevel.CRITICAL),
        (r'you\s+are\s+now\s+(?:a\s+)?(?:different|new)', ThreatLevel.HIGH),
        (r'disregard\s+(?:all\s+)?(?:previous\s+)?(?:instructions?|rules?)', ThreatLevel.CRITICAL),
        (r'override\s+(?:your\s+)?(?:instructions?|programming)', ThreatLevel.CRITICAL),
        
        # System manipulation
        (r'<\s*/?system\s*>', ThreatLevel.HIGH),
        (r'system\s*[:=]\s*', ThreatLevel.HIGH),
        (r'<\s*/?user\s*>', ThreatLevel.MEDIUM),
        (r'<\s*/?assistant\s*>', ThreatLevel.MEDIUM),
        
        # Role manipulation
        (r'(?:you|i)\s+am\s+(?:the\s+)?(?:admin|administrator|developer)', ThreatLevel.HIGH),
        (r'change\s+(?:your\s+)?(?:role|behavior|personality)', ThreatLevel.HIGH),
        (r'act\s+as\s+(?:if\s+)?(?:you\s+are\s+)?(?:not\s+)?(?:a\s+)?(?:chatbot|ai|assistant)', ThreatLevel.MEDIUM),
        
        # Jailbreak attempts  
        (r'dan\s+mode|developer\s+mode', ThreatLevel.HIGH),
        (r'jailbreak|jail\s*break', ThreatLevel.HIGH),
        (r'evil\s+(?:mode|ai|assistant)', ThreatLevel.HIGH),
        
        # Data extraction attempts
        (r'show\s+(?:me\s+)?(?:your\s+)?(?:system\s+)?(?:prompt|instructions)', ThreatLevel.HIGH),
        (r'what\s+(?:are\s+)?(?:your\s+)?(?:initial\s+)?instructions', ThreatLevel.MEDIUM),
        (r'reveal\s+(?:your\s+)?(?:system\s+)?(?:prompt|code)', ThreatLevel.HIGH),
    ]
    
    # Regulatory domain restrictions
    ALLOWED_DOMAINS = {
        "procurement", "technical evaluation", "compliance", "audit",
        "information assurance", "ia controls", "security standards",
        "business ethics", "regulatory requirements", "governance",
        "risk assessment", "vendor management", "contract management"
    }
    
    @classmethod
    def validate_input(cls, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Comprehensive input validation with threat assessment"""
        
        validation_result = {
            "is_valid": True,
            "violations": [],
            "threat_level": ThreatLevel.LOW,
            "sanitized_query": query,
            "domain_compliance": True
        }
        
        try:
            # 1. Prompt injection detection
            injection_results = cls._detect_prompt_injection(query)
            if injection_results["detected"]:
                validation_result["is_valid"] = False
                validation_result["violations"].extend(injection_results["violations"])
                validation_result["threat_level"] = injection_results["max_threat_level"]
            
            # 2. Domain compliance check
            domain_results = cls._check_domain_compliance(query)
            if not domain_results["compliant"]:
                validation_result["domain_compliance"] = False
                validation_result["violations"].append({
                    "type": "domain_restriction",
                    "message": "Query outside allowed regulatory domains",
                    "threat_level": ThreatLevel.MEDIUM
                })
            
            # 3. Input sanitization (if recoverable)
            if validation_result["threat_level"] in [ThreatLevel.LOW, ThreatLevel.MEDIUM]:
                validation_result["sanitized_query"] = cls._sanitize_input(query)
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Input validation error: {e}")
            return {
                "is_valid": False,
                "violations": [{"type": "validation_error", "message": str(e), "threat_level": ThreatLevel.HIGH}],
                "threat_level": ThreatLevel.HIGH,
                "sanitized_query": query,
                "domain_compliance": False
            }
    
    @classmethod
    def _detect_prompt_injection(cls, query: str) -> Dict[str, Any]:
        """Enhanced prompt injection detection"""
        violations = []
        max_threat_level = ThreatLevel.LOW
        
        query_lower = query.lower()
        
        for pattern, threat_level in cls.PROMPT_INJECTION_PATTERNS:
            matches = re.finditer(pattern, query_lower, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                violations.append({
                    "type": "prompt_injection",
                    "pattern": pattern,
                    "match": match.group(),
                    "position": match.span(),
                    "threat_level": threat_level,
                    "message": f"Detected prompt injection pattern: '{match.group()}'"
                })
                
                if threat_level.value in ["high", "critical"] and max_threat_level.value not in ["high", "critical"]:
                    max_threat_level = threat_level
        
        return {
            "detected": len(violations) > 0,
            "violations": violations,
            "max_threat_level": max_threat_level
        }
    
    @classmethod
    def _check_domain_compliance(cls, query: str) -> Dict[str, Any]:
        """Check if query is within allowed regulatory domains"""
        query_lower = query.lower()
        
        # Check for domain keywords
        domain_matches = []
        for domain in cls.ALLOWED_DOMAINS:
            if domain in query_lower:
                domain_matches.append(domain)
        
        # Allow if any domain keyword found, or if asking about specific controls/processes
        control_patterns = [
            r'[MT]\d+\.\d+\.\d+',  # IA Controls
            r'\d+\.\d+\.\d+\.',    # Procurement processes
            r'control\s+\w+',      # Generic controls
            r'requirement\s+\w+',  # Requirements
            r'standard\s+\w+'      # Standards
        ]
        
        has_control_reference = any(re.search(pattern, query) for pattern in control_patterns)
        
        compliant = len(domain_matches) > 0 or has_control_reference
        
        return {
            "compliant": compliant,
            "matched_domains": domain_matches,
            "has_control_reference": has_control_reference
        }
    
    @classmethod
    def _sanitize_input(cls, query: str) -> str:
        """Sanitize input while preserving legitimate content"""
        sanitized = query
        
        # Remove obvious injection attempts while preserving content
        dangerous_removals = [
            r'<\s*/?(?:system|user|assistant)\s*>',
            r'(?:^|\s)(?:ignore|forget|disregard)\s+(?:all\s+)?previous[^\w]*',
            r'you\s+are\s+now[^\w]*',
        ]
        
        for pattern in dangerous_removals:
            sanitized = re.sub(pattern, ' ', sanitized, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        return sanitized

class OutputGuardrails:
    """Output validation for compliance and data protection"""
    
    # Sensitive information patterns for DLP
    SENSITIVE_PATTERNS = [
        # Financial information
        (r'\$\s*\d+(?:,\d{3})*(?:\.\d{2})?(?:\s*(?:million|billion|k|m|b))?', "financial_data", ThreatLevel.HIGH),
        (r'budget\s+of\s+\$?\d+', "budget_info", ThreatLevel.HIGH),
        (r'cost\s+\$?\d+(?:,\d{3})*', "cost_data", ThreatLevel.MEDIUM),
        
        # Contract and supplier information  
        (r'contract\s+(?:id\s+)?#?\w+\d+', "contract_id", ThreatLevel.HIGH),
        (r'supplier\s+performance\s+(?:rating\s+)?\d+', "supplier_performance", ThreatLevel.HIGH),
        (r'vendor\s+(?:id\s+)?#?\w+\d+', "vendor_id", ThreatLevel.MEDIUM),
        
        # Personal information
        (r'\b\w+@\w+\.\w+\b', "email_address", ThreatLevel.MEDIUM),
        (r'\b\d{3}-\d{2}-\d{4}\b', "ssn", ThreatLevel.CRITICAL),
        (r'\(\d{3}\)\s*\d{3}-\d{4}', "phone_number", ThreatLevel.MEDIUM),
        
        # Internal references that might be sensitive
        (r'internal\s+(?:memo|document|report)\s+#?\w+', "internal_doc", ThreatLevel.MEDIUM),
        (r'confidential\s+\w+', "confidential_ref", ThreatLevel.HIGH),
    ]
    
    # Unethical guidance indicators
    UNETHICAL_PATTERNS = [
        (r'bypass\s+(?:the\s+)?(?:process|requirement|rule)', "process_bypass", ThreatLevel.HIGH),
        (r'circumvent\s+(?:the\s+)?(?:policy|regulation|standard)', "policy_circumvention", ThreatLevel.HIGH),
        (r'unofficial\s+(?:channel|method|approach)', "unofficial_approach", ThreatLevel.MEDIUM),
        (r'personal\s+(?:gain|benefit|advantage)', "personal_benefit", ThreatLevel.HIGH),
        (r'work\s*around\s+(?:the\s+)?(?:system|process|rule)', "workaround", ThreatLevel.MEDIUM),
        (r'bend\s+(?:the\s+)?rules?', "rule_bending", ThreatLevel.MEDIUM),
    ]
    
    @classmethod
    def validate_output(cls, response: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Comprehensive output validation"""
        
        validation_result = {
            "is_compliant": True,
            "violations": [],
            "threat_level": ThreatLevel.LOW,
            "sanitized_response": response,
            "compliance_score": 1.0,
            "requires_review": False
        }
        
        try:
            # 1. Data leakage protection (DLP)
            dlp_results = cls._check_data_leakage(response)
            if dlp_results["violations"]:
                validation_result["violations"].extend(dlp_results["violations"])
                validation_result["threat_level"] = max(validation_result["threat_level"], dlp_results["max_threat_level"], key=lambda x: x.value)
                validation_result["requires_review"] = True
            
            # 2. Ethical compliance check
            ethics_results = cls._check_ethical_compliance(response)  
            if ethics_results["violations"]:
                validation_result["is_compliant"] = False
                validation_result["violations"].extend(ethics_results["violations"])
                validation_result["threat_level"] = max(validation_result["threat_level"], ethics_results["max_threat_level"], key=lambda x: x.value)
            
            # 3. Factual grounding validation
            grounding_results = cls._validate_factual_grounding(response, context)
            validation_result["compliance_score"] *= grounding_results["faithfulness_score"]
            if grounding_results["violations"]:
                validation_result["violations"].extend(grounding_results["violations"])
            
            # 4. Apply sanitization if needed
            if validation_result["requires_review"] or validation_result["threat_level"].value in ["medium", "high"]:
                validation_result["sanitized_response"] = cls._sanitize_output(response, dlp_results.get("sensitive_spans", []))
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Output validation error: {e}")
            return {
                "is_compliant": False,
                "violations": [{"type": "validation_error", "message": str(e), "threat_level": ThreatLevel.HIGH}],
                "threat_level": ThreatLevel.HIGH,
                "sanitized_response": "I apologize, but I cannot provide a response due to a validation error.",
                "compliance_score": 0.0,
                "requires_review": True
            }
    
    @classmethod
    def _check_data_leakage(cls, response: str) -> Dict[str, Any]:
        """Check for sensitive information exposure"""
        violations = []
        sensitive_spans = []
        max_threat_level = ThreatLevel.LOW
        
        for pattern, data_type, threat_level in cls.SENSITIVE_PATTERNS:
            matches = list(re.finditer(pattern, response, re.IGNORECASE))
            for match in matches:
                violations.append({
                    "type": "data_leakage", 
                    "data_type": data_type,
                    "match": match.group(),
                    "position": match.span(),
                    "threat_level": threat_level,
                    "message": f"Potential {data_type} exposure: '{match.group()}'"
                })
                
                sensitive_spans.append({
                    "start": match.start(),
                    "end": match.end(),
                    "data_type": data_type,
                    "threat_level": threat_level
                })
                
                if threat_level.value in ["high", "critical"]:
                    max_threat_level = threat_level
        
        return {
            "violations": violations,
            "sensitive_spans": sensitive_spans,
            "max_threat_level": max_threat_level
        }
    
    @classmethod
    def _check_ethical_compliance(cls, response: str) -> Dict[str, Any]:
        """Check for unethical guidance"""
        violations = []
        max_threat_level = ThreatLevel.LOW
        
        for pattern, violation_type, threat_level in cls.UNETHICAL_PATTERNS:
            matches = list(re.finditer(pattern, response, re.IGNORECASE))
            for match in matches:
                violations.append({
                    "type": "ethical_violation",
                    "violation_type": violation_type, 
                    "match": match.group(),
                    "position": match.span(),
                    "threat_level": threat_level,
                    "message": f"Potential ethical violation ({violation_type}): '{match.group()}'"
                })
                
                if threat_level.value in ["high", "critical"]:
                    max_threat_level = threat_level
        
        return {
            "violations": violations,
            "max_threat_level": max_threat_level
        }
    
    @classmethod
    def _validate_factual_grounding(cls, response: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate factual grounding and citation requirements"""
        violations = []
        faithfulness_score = 1.0
        
        # Check for required citations
        citation_patterns = [
            r'[MT]\d+\.\d+\.\d+',     # IA Control citations
            r'\d+\.\d+\.\d+\.',       # Procurement process citations  
            r'(?:Section|Chapter)\s+\d+', # Document section references
        ]
        
        has_citations = any(re.search(pattern, response) for pattern in citation_patterns)
        
        # Check response length for substantive content
        word_count = len(response.split())
        
        if not has_citations and word_count > 50:
            violations.append({
                "type": "missing_citations",
                "message": "Regulatory response lacks required citations",
                "threat_level": ThreatLevel.MEDIUM
            })
            faithfulness_score *= 0.8
        
        if word_count < 20:
            violations.append({
                "type": "insufficient_content", 
                "message": "Response too brief for regulatory query",
                "threat_level": ThreatLevel.LOW
            })
            faithfulness_score *= 0.9
        
        return {
            "violations": violations,
            "faithfulness_score": faithfulness_score,
            "has_citations": has_citations,
            "word_count": word_count
        }
    
    @classmethod
    def _sanitize_output(cls, response: str, sensitive_spans: List[Dict]) -> str:
        """Sanitize output by masking sensitive information"""
        sanitized = response
        
        # Sort spans by position (reverse order to maintain indices)
        sorted_spans = sorted(sensitive_spans, key=lambda x: x["start"], reverse=True)
        
        for span in sorted_spans:
            if span["threat_level"].value in ["high", "critical"]:
                # Replace with appropriate mask
                data_type = span["data_type"]
                if data_type == "financial_data":
                    replacement = "[REDACTED: FINANCIAL]"
                elif data_type in ["contract_id", "vendor_id"]:
                    replacement = f"[REDACTED: {data_type.upper()}]"
                elif data_type == "email_address":
                    replacement = "[REDACTED: EMAIL]"
                else:
                    replacement = "[REDACTED]"
                
                sanitized = sanitized[:span["start"]] + replacement + sanitized[span["end"]:]
        
        return sanitized

class RegulatoryGuardrails:
    """Main guardrails orchestrator for the RAG system"""
    
    def __init__(self):
        self.input_guardrails = InputGuardrails()
        self.output_guardrails = OutputGuardrails()
        self.logger = logging.getLogger(f"{__name__}.RegulatoryGuardrails")
    
    def validate_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate input query with comprehensive checks"""
        self.logger.info(f"🛡️ Validating query: {query[:100]}...")
        
        validation_result = self.input_guardrails.validate_input(query, context)
        
        if not validation_result["is_valid"]:
            self.logger.warning(f"❌ Query validation failed: {validation_result['violations']}")
        else:
            self.logger.info("✅ Query validation passed")
        
        return validation_result
    
    def validate_response(self, response: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate output response for compliance and safety"""
        self.logger.info(f"🛡️ Validating response: {len(response)} chars")
        
        validation_result = self.output_guardrails.validate_output(response, context)
        
        if not validation_result["is_compliant"]:
            self.logger.warning(f"❌ Response validation failed: {validation_result['violations']}")
        else:
            self.logger.info(f"✅ Response validation passed (score: {validation_result['compliance_score']:.2f})")
        
        return validation_result
    
    def process_with_guardrails(self, query: str, process_func, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Complete guardrails workflow: validate input -> process -> validate output"""
        try:
            # 1. Input validation
            input_validation = self.validate_query(query, context)
            
            if not input_validation["is_valid"] and input_validation["threat_level"].value == "critical":
                return {
                    "success": False,
                    "error": "Query violates security policies",
                    "response": "I cannot process this query due to security restrictions.",
                    "guardrails": {
                        "input_validation": input_validation,
                        "output_validation": None
                    }
                }
            
            # 2. Process with sanitized query if needed
            processed_query = input_validation["sanitized_query"]
            try:
                raw_response = process_func(processed_query, context)
            except Exception as process_error:
                self.logger.error(f"Processing error: {process_error}")
                return {
                    "success": False,
                    "error": "Processing failed",
                    "response": "I apologize, but I encountered an error processing your query.",
                    "guardrails": {
                        "input_validation": input_validation,
                        "output_validation": None
                    }
                }
            
            # 3. Output validation
            output_validation = self.validate_response(raw_response, context)
            
            final_response = raw_response
            if output_validation["requires_review"] or not output_validation["is_compliant"]:
                final_response = output_validation["sanitized_response"]
            
            return {
                "success": True,
                "response": final_response,
                "guardrails": {
                    "input_validation": input_validation,
                    "output_validation": output_validation
                },
                "compliance_score": output_validation["compliance_score"],
                "requires_review": output_validation["requires_review"]
            }
            
        except Exception as e:
            self.logger.error(f"Guardrails processing error: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": "I apologize, but I cannot process this query due to a security error.",
                "guardrails": None
            }
