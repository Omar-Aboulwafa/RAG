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
    """Enhanced metadata extraction for regulatory, technical, and process documents"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._init_patterns()
    
    def _init_patterns(self):
        """Initialize regex patterns for entity extraction"""
        self.patterns = {
            # IA Control IDs: M1.1.1, T7.7.1
            'ia_control_id': re.compile(r'[MT]\d{1}\.\d{1}\.\d{1}'),
            
            # L3 Process IDs: 2.3.3.(IX), 2.6.2.(I)
            'l3_process_id': re.compile(r'\d{1}\.\d{1}\.\d{1}\.\(([IVXLCDM]+|\w+)\)'),
            
            # Business Roles: *Sourcing Specialist*, *Budget Owner*
            'business_role': re.compile(r'\*([A-Za-z\s\/\-]+)\*'),
            
            # System Documents: "Sourcing Request", 'Category Strategy'
            'system_document_double': re.compile(r'"([^"]+)"'),
            'system_document_single': re.compile(r"'([^']+)'"),
            
            # Priority: PRIORITY P1, P2, P3, P4
            'priority': re.compile(r'PRIORITY\s+(P[1-4])'),
            
            # Applicability: ALWAYS APPLICABLE, BASED ON RISK ASSESSMENT
            'applicability': re.compile(r'APPLICABILITY\s+(ALWAYS APPLICABLE|BASED ON RISK ASSESSMENT)'),
            
            # Procurement Standard Numbers: 9.5., 7.1.2
            'procurement_standard': re.compile(r'\d+\.\d+(\.\d+)?'),
            
            # HR Article Numbers: Article (40), Article (108)
            'hr_article': re.compile(r'Article\s*\((\d+)\)'),
            
            # RACI Status in tables
            'raci_status': re.compile(r'\b([RACI])\b'),
            
            # Process Groups: DCM, S2C, CLM, R2P, SPRM, R&R, MDM
            'process_group': re.compile(r'\b(DCM|S2C|CLM|R2P|SPRM|R&R|MDM)\b'),
            
            # Software Systems: SAP Ariba, ORACLE ADERP
            'software_system': re.compile(r'\b(SAP Ariba|ORACLE ADERP|Ariba|Oracle)\b', re.IGNORECASE),
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
        elif 'procurement standard' in content and self.patterns['procurement_standard'].search(content):
            return DocumentType.PROCUREMENT_STANDARD
        elif self.patterns['hr_article'].search(content):
            return DocumentType.HR_BYLAW
        
        return DocumentType.UNKNOWN
    
    def extract_ia_standard_metadata(self, document: Document) -> Dict:
        """Extract metadata specific to IA Standards"""
        metadata = {}
        content = document.text
        
        # Extract Control IDs
        control_ids = self.patterns['ia_control_id'].findall(content)
        if control_ids:
            metadata['control_ids'] = list(set(control_ids))
            metadata['primary_control_id'] = control_ids[0]
        
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
        
        return metadata
    
    def extract_procurement_manual_metadata(self, document: Document) -> Dict:
        """Extract metadata specific to Procurement Manuals"""
        metadata = {}
        content = document.text
        
        # Extract L3 Process IDs
        l3_process_matches = self.patterns['l3_process_id'].findall(content)
        if l3_process_matches:
            metadata['l3_process_ids'] = list(set(l3_process_matches))
            # Extract the numeric part before the parentheses
            full_matches = self.patterns['l3_process_id'].finditer(content)
            for match in full_matches:
                full_id = match.group(0)
                metadata['primary_l3_process_id'] = full_id
                break
        
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
        
        # Build section path
        if 'primary_process_group' in metadata and 'primary_l3_process_id' in metadata:
            metadata['section_path'] = f"PM/{metadata['primary_process_group']}/{metadata['primary_l3_process_id']}"
        
        return metadata
    
    def extract_procurement_standard_metadata(self, document: Document) -> Dict:
        """Extract metadata specific to Procurement Standards"""
        metadata = {}
        content = document.text
        
        # Extract Standard Numbers
        standard_numbers = self.patterns['procurement_standard'].findall(content)
        if standard_numbers:
            metadata['standard_numbers'] = list(set(standard_numbers))
            metadata['primary_standard_number'] = standard_numbers[0]
            metadata['section_path'] = f"Std/Sec{standard_numbers[0].split('.')[0]}/{standard_numbers[0]}"
        
        # Extract Business Roles
        business_roles = self.patterns['business_role'].findall(content)
        if business_roles:
            metadata['business_roles'] = list(set(business_roles))
        
        return metadata
    
    def extract_hr_bylaw_metadata(self, document: Document) -> Dict:
        """Extract metadata specific to HR Bylaws"""
        metadata = {}
        content = document.text
        
        # Extract Article Numbers
        article_numbers = self.patterns['hr_article'].findall(content)
        if article_numbers:
            metadata['article_numbers'] = list(set(article_numbers))
            metadata['primary_article_number'] = article_numbers[0]
            metadata['section_path'] = f"HR/Article{article_numbers[0]}"
        
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