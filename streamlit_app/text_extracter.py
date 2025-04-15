import PyPDF2
import tempfile
import os
import re

def extract_text_from_pdf(pdf_file):
    
    """
    Extract text from a PDF file
    
    Args:
        pdf_file: File object of the PDF
        
    Returns:
        str: Extracted text from the PDF
    """
    # Create a temporary file to save the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(pdf_file.getvalue())
        temp_path = temp_file.name
    
    # Open the PDF file using PyPDF2
    text = ""
    try:
        with open(temp_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Get the number of pages
            num_pages = len(pdf_reader.pages)
            
            # Extract text from each page
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"
    except Exception as e:
        text = f"Error extracting text: {str(e)}"
    finally:
        # Clean up the temporary file
        os.unlink(temp_path)
    
    return text

def extract_paper_info(pdf_text):
    """
    Extracts the title and university name from PDF text content
    
    Args:
        pdf_text (str): The text content extracted from a PDF
        
    Returns:
        str: Formatted output with title and university information
    """
    # Common patterns for paper titles and university affiliations
    
    # Look for title - often at the beginning, sometimes preceded by keywords
    # Titles are usually in larger font, centered, or followed by author names
    print("len of pdf text", len(pdf_text))
    title_pattern = r"^([^\n.]{10,150})\n"  # Simple approach: first non-short line
    
    # More sophisticated approaches might look for specific markers
    sophisticated_title_patterns = [
        r"TITLE[:\s]+([^\n]+)",  # Explicit "TITLE:" marker
        r"^[\s\*]*([A-Z][^.!?\n]{10,150})\n",  # Capitalized first line that's reasonably long
        r"^\s*([^\n.]{20,150})\n\s*(?:by|authors?|submitted by)"  # Title followed by author indicator
    ]
    
    # Look for university name
    # Universities often appear after author names, with specific keywords
    university_patterns = [
        r"University\s+of\s+[A-Za-z\s,]+",  # University of X
        r"([A-Za-z\s]+University[A-Za-z\s,]*)",  # X University
        r"([A-Za-z\s]+Institute\s+of\s+Technology[A-Za-z\s,]*)",  # X Institute of Technology
        r"([A-Za-z\s]+College[A-Za-z\s,]*)",  # X College
        r"Department\s+of\s+[A-Za-z\s,]+,\s+([^,\n]+)",  # Department of X, [University]
        r"Affiliation[s]?[:\s]+([^,\n]+)"  # Explicit "Affiliation:" marker
    ]
    
    # Extract title
    title = ""
    
    # Try simple pattern first
    title_match = re.search(title_pattern, pdf_text, re.MULTILINE)
    if title_match and title_match.group(1):
        title = title_match.group(1).strip()
    else:
        # Try more sophisticated patterns
        for pattern in sophisticated_title_patterns:
            title_match = re.search(pattern, pdf_text, re.MULTILINE | re.IGNORECASE)
            if title_match and title_match.group(1):
                title = title_match.group(1).strip()
                break
    
    university = "Unknown University"
    for pattern in university_patterns:
        uni_match = re.search(pattern, pdf_text, re.MULTILINE | re.IGNORECASE)
        if uni_match and uni_match.group(0):
            university = uni_match.group(0).strip()
            break
    
    return f"This paper is about {title}. It is developed by researchers from {university}"


