# src/tools.py - Custom Tools for Document Processing\n\"""
from crewai_tools import BaseTool
import pypdf\nfrom typing import List
from langchain_community.document_loaders import PyPDFLoader\nfrom pathlib import Path\n\nclass ComplianceCheckerTool(BaseTool):\n    """A tool designed to read and search content within compliance documents or the target document.
\n    This tool abstracts PDF loading logic, making it callable by agents.\n    """
    name = "Compliance Document Reader"
    description = "Useful for reading text content from uploaded PDF files (knowledge base or target document). Takes a file path and identifies the text."
    args_schema: dict = None # You might need to explicitly define an args schema if passing more than one arg type
\n    def __init__(self, knowledge_base_docs: List[Path] = None):\n        """Initialize the tool with a set of reference documents.

        Args:\n            knowledge_base_docs: A list of pathlib.Path objects pointing to the source PDFs.\n        """
        self.knowledge_base_paths = knowledge_base_docs if knowledge_base_docs else []\n        print(f\"Tool initialized with {len(self.knowledge_base_paths)} knowledge base documents.\")
\n    def _run(self, file_path: str) -> str:\n        """Loads a specified PDF and extracts all text content.
        The 'file_path' should point to either the target document or one of the loaded knowledge bases.\n        """
        try:\n            pdf_path = Path(file_path)\n            if not pdf_path.exists():\n                return f\"Error: File not found at {file_path}\"
\n            print(f\"Loading PDF from: {pdf_path}...\")
            loader = PyPDFLoader(str(pdf_path))\n            docs = loader.load()\n            
            # Concatenate text from all pages/chunks into one string for the agent to process easily\n            full_text = "\n---\n".join([doc.page_content for doc in docs])\n            return full_text\n        except Exception as e:\n            return f"Error reading PDF: {e}"

    async def _arun(self, file_path: str) -> str:\n        # For async support if needed later\n        return self._run(file_path)