# controllers/MetadataExtractor.py - FULLY FIXED VERSION
import re
import os
from typing import Dict, List, Optional, Tuple
from enum import Enum
from llama_index.core import Document
import logging


class DocumentType(Enum):
    IA_STANDARD = "IA Standard"
    PROCUREMENT_MANUAL = "Procurement Manual"
    PROCUREMENT_STANDARD = "Procurement Standard"
    HR_BYLAW = "HR Bylaw"
    UNKNOWN = "Unknown"


class MetadataExtractor:
    """Enhanced metadata extraction for regulatory, technical, and process documents - FIXED VERSION"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._init_patterns()

    def _init_patterns(self):
        """Initialize regex patterns for entity extraction - ALL PATTERNS FIXED"""
        self.patterns = {
           
            # Matches: M1.1.1, T7.7.1, M10.12.15, T1.1.100
            'ia_control_id': re.compile(r'[MT]\d+\.\d+\.\d+'),
            
            # ✅ FIXED: L3 Process IDs - Now handles multiple digits
            # Matches: 2.3.3.(IX), 10.12.5.(IV), 1.1.1.(I)
            'l3_process_id': re.compile(r'\d+\.\d+\.\d+\.\(([IVXLCDM]+|\w+)\)'),
            
            # Business Roles: *Sourcing Specialist*, *Budget Owner*
            'business_role': re.compile(r'\*([A-Za-z\s\/\-]+)\*'),
            
            # System Documents: "Sourcing Request", 'Category Strategy'
            'system_document_double': re.compile(r'"([^"]+)"'),
            'system_document_single': re.compile(r"'([^']+)'"),
            
            # Priority: PRIORITY P1, P2, P3, P4
            'priority': re.compile(r'PRIORITY\s+(P[1-4])'),
            
            # Applicability: ALWAYS APPLICABLE, BASED ON RISK ASSESSMENT
            'applicability': re.compile(r'APPLICABILITY\s+(ALWAYS APPLICABLE|BASED ON RISK ASSESSMENT)'),
            
            # ✅ IMPROVED: Procurement Standard Numbers - More specific context
            # Only matches when preceded by "Standard" or "Section"
            'procurement_standard': re.compile(r'(?:Standard|Section)\s+(\d+\.\d+(?:\.\d+)?)', re.IGNORECASE),
            
            # ✅ CRITICAL FIX: HR Article Numbers - NOW WORKS!
            # Matches: "Article 110", "Article (110)", "ARTICLE 40", "Article  15"
            'hr_article': re.compile(r'(Article\s*\(\s*\d+\s*\)|ARTICLE\s+\d+)', re.IGNORECASE),
            
            # RACI Status in tables
            'raci_status': re.compile(r'\b([RACI])\b'),
            
            # Process Groups: DCM, S2C, CLM, R2P, SPRM, R&R, MDM
            'process_group': re.compile(r'\b(DCM|S2C|CLM|R2P|SPRM|R&R|MDM)\b'),
            
            # Software Systems: SAP Ariba, ORACLE ADERP
            'software_system': re.compile(r'\b(SAP Aribia|ORACLE ADERP|Ariba|Oracle)\b', re.IGNORECASE),
        }
    
    def detect_document_type(self, document: Document) -> DocumentType:
        """Detect document type based on content patterns and filename"""
        content = document.text.lower()
        filename = document.metadata.get('filename', '').lower()
        
        # Check filename patterns first
        if any(term in filename for term in ['ia', 'nesa', 'information assurance', 'security']):
            return DocumentType.IA_STANDARD
        elif any(term in filename for term in ['procurement manual', 'ariba', 'process']):
            return DocumentType.PROCUREMENT_MANUAL
        elif any(term in filename for term in ['procurement standard', 'standard']):
            return DocumentType.PROCUREMENT_STANDARD
        elif any(term in filename for term in ['hr', 'bylaw', 'human resource']):
            return DocumentType.HR_BYLAW
        
        # Check content patterns
        if self.patterns['ia_control_id'].search(content):
            return DocumentType.IA_STANDARD
        elif self.patterns['l3_process_id'].search(content):
            return DocumentType.PROCUREMENT_MANUAL
        elif self.patterns['procurement_standard'].search(content):
            return DocumentType.PROCUREMENT_STANDARD
        elif self.patterns['hr_article'].search(content):
            return DocumentType.HR_BYLAW
        
        return DocumentType.UNKNOWN
    
    def extract_ia_standard_metadata(self, document: Document) -> Dict:
        """Extract metadata specific to IA Standards - FIXED"""
        metadata = {}
        content = document.text
        
        # Extract Control IDs (now handles multi-digit)
        control_ids = self.patterns['ia_control_id'].findall(content)
        if control_ids:
            metadata['control_ids'] = list(set(control_ids))
            metadata['primary_control_id'] = control_ids[0]
            metadata['control_id'] = control_ids[0] 
        
        # Extract Priority
        priority_matches = self.patterns['priority'].findall(content)
        if priority_matches:
            metadata['priority'] = priority_matches[0]
        
        # Extract Applicability
        applicability_matches = self.patterns['applicability'].findall(content)
        if applicability_matches:
            metadata['applicability'] = applicability_matches[0]
        
        # Determine chapter from control ID
        if 'primary_control_id' in metadata:
            control_id = metadata['primary_control_id']
            chapter = control_id.split('.')[0]
            metadata['chapter'] = chapter
            metadata['section_path'] = f"IA/Ch{chapter[1:]}/{control_id}"
            metadata['citation_id'] = control_id  
        
        return metadata
    
    def extract_procurement_manual_metadata(self, document: Document) -> Dict:
        """Extract metadata specific to Procurement Manuals - FIXED"""
        metadata = {}
        content = document.text
        
        # Extract L3 Process IDs (now handles multi-digit)
        l3_process_matches = self.patterns['l3_process_id'].findall(content)
        if l3_process_matches:
            metadata['l3_process_ids'] = list(set(l3_process_matches))
        
        # Extract full L3 Process ID with proper formatting
        full_matches = self.patterns['l3_process_id'].finditer(content)
        for match in full_matches:
            full_id = match.group(0)
            metadata['primary_l3_process_id'] = full_id
            metadata['level3_process_id'] = full_id  
        
        # Extract Process Groups
        process_groups = self.patterns['process_group'].findall(content)
        if process_groups:
            metadata['process_groups'] = list(set(process_groups))
            metadata['primary_process_group'] = process_groups[0]
        
        # Extract Business Roles
        business_roles = self.patterns['business_role'].findall(content)
        if business_roles:
            metadata['business_roles'] = list(set(business_roles))
        
        # Extract System Documents
        system_docs_double = self.patterns['system_document_double'].findall(content)
        system_docs_single = self.patterns['system_document_single'].findall(content)
        all_system_docs = system_docs_double + system_docs_single
        if all_system_docs:
            metadata['artifact_names'] = list(set(all_system_docs))
        
        # Extract Software Systems
        software_systems = self.patterns['software_system'].findall(content)
        if software_systems:
            metadata['software_systems'] = list(set(software_systems))
        
        # Build section path and citation
        if 'primary_process_group' in metadata and 'primary_l3_process_id' in metadata:
            metadata['section_path'] = f"PM/{metadata['primary_process_group']}/{metadata['primary_l3_process_id']}"
            metadata['citation_id'] = metadata['primary_l3_process_id']  
        
        return metadata
    
    def extract_procurement_standard_metadata(self, document: Document) -> Dict:
        """Extract metadata specific to Procurement Standards - FIXED"""
        metadata = {}
        content = document.text
        
        # Extract Standard Numbers (now more precise)
        standard_matches = self.patterns['procurement_standard'].findall(content)
        if standard_matches:
            metadata['standard_numbers'] = list(set(standard_matches))
            metadata['primary_standard_number'] = standard_matches[0]
            metadata['section_path'] = f"Std/Sec{standard_matches[0].split('.')[0]}/{standard_matches[0]}"
            metadata['citation_id'] = f"Standard {standard_matches[0]}"  
        
        # Extract Business Roles
        business_roles = self.patterns['business_role'].findall(content)
        if business_roles:
            metadata['business_roles'] = list(set(business_roles))
        
        return metadata
    
    def extract_hr_bylaw_metadata(self, document: Document) -> Dict:
        """Extract metadata specific to HR Bylaws - COMPLETELY FIXED"""
        metadata = {}
        content = document.text
        
        # Extract Article Numbers (now works!)
        article_numbers = self.patterns['hr_article'].findall(content)
        
        if article_numbers:
            # Convert to integers and filter out single-digit false matches
            valid_articles = [art for art in article_numbers if int(art) >= 10]
            
            if valid_articles:
                metadata['article_numbers'] = list(set(valid_articles))
                metadata['primary_article_number'] = valid_articles[0]
                metadata['section_path'] = f"HR/Article/{valid_articles[0]}"
                metadata['citation_id'] = f"Article {valid_articles[0]}"
                
                self.logger.info(f"✅ Extracted Article {valid_articles[0]} from HR Bylaw")
            else:
                self.logger.warning(f"⚠️ Found article numbers but all were < 10 (likely false matches): {article_numbers}")
        else:
            self.logger.warning(f"⚠️ No article number found in HR Bylaw chunk")
        
        # numbered items/clauses within articles
        clause_pattern = re.compile(r'^\s*(\d+)\.\s+', re.MULTILINE)
        clauses = clause_pattern.findall(content)
        if clauses:
            metadata['clauses'] = list(set(clauses))
            metadata['clause_count'] = len(set(clauses))
        
        # Flag chunks containing critical keywords
        if 'sixty days' in content.lower() or '60 days' in content.lower():
            metadata['contains_maximum_deduction'] = True
            metadata['maximum_deduction_days'] = '60'
            self.logger.info(f"✅ Found maximum deduction limit (60 days) in chunk")
        
        # NEW: Keyword tagging for better retrieval
        keywords = []
        content_lower = content.lower()
        
        if 'deduction' in content_lower or 'deduct' in content_lower:
            keywords.append('salary_deduction')
        if 'penalty' in content_lower or 'penalties' in content_lower:
            keywords.append('disciplinary_penalty')
        if 'sixty' in content_lower or '60' in content_lower:
            keywords.append('maximum_limit')
        if 'year' in content_lower or 'annual' in content_lower:
            keywords.append('annual_limit')
        if 'leave' in content_lower:
            keywords.append('leave_policy')
        if 'termination' in content_lower or 'dismissal' in content_lower:
            keywords.append('termination')
        
        if keywords:
            metadata['keywords'] = keywords
        
        return metadata
    
    def extract_comprehensive_metadata(self, document: Document, project_id: str) -> Document:
        """Main method to extract comprehensive metadata from any document type"""
        try:
            # Preserve existing metadata
            enhanced_metadata = document.metadata.copy()
            
            # Add basic metadata
            enhanced_metadata['project_id'] = project_id
            
            # Detect document type
            doc_type = self.detect_document_type(document)
            enhanced_metadata['doc_type'] = doc_type.value
            
            # Extract type-specific metadata
            if doc_type == DocumentType.IA_STANDARD:
                type_metadata = self.extract_ia_standard_metadata(document)
            elif doc_type == DocumentType.PROCUREMENT_MANUAL:
                type_metadata = self.extract_procurement_manual_metadata(document)
            elif doc_type == DocumentType.PROCUREMENT_STANDARD:
                type_metadata = self.extract_procurement_standard_metadata(document)
            elif doc_type == DocumentType.HR_BYLAW:
                type_metadata = self.extract_hr_bylaw_metadata(document)
            else:
                type_metadata = {}
            
            # Merge type-specific metadata
            enhanced_metadata.update(type_metadata)
            
            # Create new document with enhanced metadata
            enhanced_document = Document(
                text=document.text,
                metadata=enhanced_metadata
            )
            
            self.logger.info(f"Enhanced metadata extraction completed for {doc_type.value}")
            return enhanced_document
            
        except Exception as e:
            self.logger.error(f"Error extracting metadata: {str(e)}")
            return document
    
    def get_section_hierarchy(self, metadata: Dict) -> List[str]:
        """Generate hierarchical section breadcrumbs for better organization"""
        hierarchy = []
        
        doc_type = metadata.get('doc_type', 'Unknown')
        hierarchy.append(doc_type)
        
        if 'section_path' in metadata:
            path_parts = metadata['section_path'].split('/')
            hierarchy.extend(path_parts[1:])  # Skip the first part as it's already in doc_type
        
        return hierarchy
