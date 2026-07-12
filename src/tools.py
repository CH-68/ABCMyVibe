from crewai.tools import tool
from pypdf import PdfReader

@tool("Read Target Document Section by Section")
def read_target_document_by_page(file_path: str, page_number: int) -> str:
    """
    Reads a specific page number from the target document to ensure a 
    meticulous, iterative, page-by-page compliance review. 
    Use this to pull text page-by-page sequentially starting from page 1.
    """
    try:
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)
        
        if page_number < 1 or page_number > total_pages:
            return f"Error: Page {page_number} out of bounds. Total pages: {total_pages}."
            
        page_text = reader.pages[page_number - 1].extract_text()
        return f"--- START OF PAGE {page_number}/{total_pages} ---\n{page_text}\n--- END OF PAGE {page_number} ---"
    except Exception as e:
        return f"Error reading file: {str(e)}"