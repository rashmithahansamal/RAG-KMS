from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import  PyPDFLoader
import pandas as pd

class ResearchPaperAnalyzer:
    def __init__(self, llm_service, vector_store):
        self.llm_service = llm_service
        self.vector_store = vector_store
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Larger chunks for research papers
            chunk_overlap=300,
            separators=["\n\n", "\n", ". ", " "]
        )
    
    def process_research_paper(self, pdf_file, paper_metadata=None):
        """Extract and analyze research paper content"""
        try:
            # Extract text from PDF
            text = self._extract_pdf_text(pdf_file)
            
            # Create structured document with metadata
            doc_metadata = {
                "document_type": "research_paper",
                "title": paper_metadata.get("title", "Unknown"),
                "authors": paper_metadata.get("authors", "Unknown"),
                "year": paper_metadata.get("year", "Unknown"),
                "journal": paper_metadata.get("journal", "Unknown")
            }
            
            # Split into sections (Abstract, Introduction, Methods, Results, etc.)
            sections = self._identify_sections(text)
            documents = []
            
            for section_name, section_text in sections.items():
                chunks = self.text_splitter.split_text(section_text)
                for i, chunk in enumerate(chunks):
                    section_metadata = doc_metadata.copy()
                    section_metadata.update({
                        "section": section_name,
                        "chunk_id": i
                    })
                    documents.append(Document(
                        page_content=chunk,
                        metadata=section_metadata
                    ))
            
            # Add to vector store
            self.vector_store.add_documents(documents)
            
            # Generate paper summary
            summary = self._generate_paper_summary(text, doc_metadata)
            
            return {
                "success": True,
                "paper_id": doc_metadata["title"],
                "sections_processed": list(sections.keys()),
                "summary": summary
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    
    
    def extract_methodology(self, paper_title):
        """Extract and explain research methodology"""
        methodology_prompt = f"""
        From the paper "{paper_title}", extract and explain:
        1. Research design and approach
        2. Data collection methods
        3. Sample size and characteristics
        4. Statistical analysis techniques used
        5. Limitations of the methodology
        6. How this methodology could be applied to similar research
        """
        
        return self.llm_service.get_course_answer(methodology_prompt, "methodology")
    
    def _extract_pdf_text(self, pdf_file):
        """Extract text from PDF file"""
        try:
            text = ""
            with open(pdf_file, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            return text
        except Exception as e:
            raise Exception(f"Error extracting PDF text: {str(e)}")
    
    def _identify_sections(self, text):
        """Identify common academic paper sections"""
        sections = {
            "abstract": "",
            "introduction": "",
            "literature_review": "",
            "methodology": "",
            "results": "",
            "discussion": "",
            "conclusion": "",
            "references": ""
        }
        
        # Simple section identification based on common headings
        text_lower = text.lower()
        section_markers = {
            "abstract": ["abstract", "summary"],
            "introduction": ["introduction", "background"],
            "literature_review": ["literature review", "related work", "previous work"],
            "methodology": ["methodology", "methods", "experimental setup", "approach"],
            "results": ["results", "findings", "experiments"],
            "discussion": ["discussion", "analysis"],
            "conclusion": ["conclusion", "conclusions", "summary"],
            "references": ["references", "bibliography"]
        }
        
        # Basic section extraction (simplified approach)
        for section_name, markers in section_markers.items():
            for marker in markers:
                if marker in text_lower:
                    # Find section text (simplified - would need more robust parsing)
                    start_idx = text_lower.find(marker)
                    if start_idx != -1:
                        # Take next 500 characters as section content
                        sections[section_name] = text[start_idx:start_idx + 500]
                        break
        
        return sections
    
    def _generate_paper_summary(self, text, metadata):
        """Generate structured summary of research paper"""
        summary_prompt = f"""
        Summarize this research paper:
        Title: {metadata['title']}
        Authors: {metadata['authors']}
        
        Text: {text[:3000]}...
        
        Provide a structured summary with:
        1. Research objective/question
        2. Key methodology
        3. Main findings
        4. Implications
        5. Limitations
        """
        
        return self.llm_service.get_course_answer(summary_prompt, "summary")