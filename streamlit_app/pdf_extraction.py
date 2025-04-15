import re
import PyPDF2
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_bytes):
    """
    Extract text from a PDF file using PyMuPDF with PyPDF2 as fallback.
    
    Args:
        pdf_bytes: PDF file as bytes
        
    Returns:
        str: Extracted text from the PDF
    """
    try:
        # Try PyMuPDF first (better extraction)
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            return text
        except Exception as e:
            print(f"PyMuPDF extraction failed, falling back to PyPDF2: {e}")
            
            # Fall back to PyPDF2 if PyMuPDF fails
            from io import BytesIO
            pdf_stream = BytesIO(pdf_bytes)
            reader = PyPDF2.PdfReader(pdf_stream)
            text = ''
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:  # Check if page extraction was successful
                    text += page_text + '\n'
            return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

def clean_text(text):
    """
    Clean the extracted text based on specified criteria.
    Only removes sections if the text exceeds 30k characters.
    
    Args:
        text (str): Raw text extracted from PDF
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    print(f"Initial text length: {len(text)} characters")
    
    # Store the original text and length
    original_text = text
    original_length = len(text)
    
    # If text is already under 30k, return it as is
    if original_length <= 30000:
        print("Text is already under 30k characters, keeping all content.")
        return text
    
    # Define patterns to selectively remove sections in order of priority
    section_patterns = [
        # 1. Contributors/authors at the beginning
        {
            'name': 'contributors',
            'pattern': r'^.*?(?:university|research|institute|lab|inc|corp|ltd|\@|\(|\))\s*\n',
            'flags': re.IGNORECASE | re.DOTALL | re.MULTILINE
        },
        # 2. Abstract
        {
            'name': 'abstract',
            'pattern': r'abstract\s*\n.*?(?=\n\s*\d+\.?\s+\w+|\n\s*[A-Z][a-z]+\s*\n|\n\s*[A-Z][A-Z\s]+\n)',
            'flags': re.IGNORECASE | re.DOTALL
        },
        # 3. References section
        {
            'name': 'references',
            'pattern': r'references\s*\n.*?$|bibliography\s*\n.*?$|works cited\s*\n.*?$',
            'flags': re.IGNORECASE | re.DOTALL
        },
        # 4. Appendix section
        {
            'name': 'appendix',
            'pattern': r'appendix\s*\n.*?$|appendix \w+\s*\n.*?$',
            'flags': re.IGNORECASE | re.DOTALL
        },
        # 5. Acknowledgments section
        {
            'name': 'acknowledgments',
            'pattern': r'acknowledgements?\s*\n.*?(?=\n\s*\d+\.?\s+\w+|\n\s*[A-Z][a-z]+\s*\n|\n\s*[A-Z][A-Z\s]+\n|$)',
            'flags': re.IGNORECASE | re.DOTALL
        },
        # 6. Citations within text
        {
            'name': 'citations',
            'pattern': r'\[\s*\d+\s*\]|\(\s*[A-Za-z]+\s*et al\.\s*,\s*\d{4}\s*\)|\[\d+(?:,\s*\d+)*\]',
            'flags': re.IGNORECASE
        },
        # 7. Email addresses
        {
            'name': 'emails',
            'pattern': r'\S+@\S+\.\S+',
            'flags': 0
        },
        # 8. Page numbers
        {
            'name': 'page_numbers',
            'pattern': r'\n\s*\d+\s*\n',
            'flags': 0,
            'replacement': '\n'
        }
    ]
    
    # Apply patterns one by one until text is under 30k
    for section in section_patterns:
        if len(text) <= 30000:
            break
            
        print(f"Removing {section['name']} to reduce text length ({len(text)} characters)")
        
        replacement = section.get('replacement', '')
        text = re.sub(section['pattern'], replacement, text, flags=section['flags'])
        
        # Check if we actually removed content
        if len(text) == len(original_text):
            print(f"No {section['name']} found to remove")
    
    # If still over 30k after removing all sections, truncate
    if len(text) > 30000:
        print(f"Still over 30k after removing sections, truncating ({len(text)} characters)")
        text = text[:30000]
    
    # If we've lost too much content, revert to truncated original
    if len(text) < 0.5 * min(original_length, 30000):
        print(f"Cleaned text too short ({len(text)} chars), reverting to truncated original")
        text = original_text[:30000]
    
    # Remove extra whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = text.strip()
    
    # Replace multiple consecutive spaces with a single space
    text = re.sub(r' +', ' ', text)
    
    print(f"Final text length: {len(text)} characters")
    return text

def process_pdf(pdf_bytes):
    """
    Process a PDF file and extract cleaned text.
    
    Args:
        pdf_bytes: PDF file content as bytes
        
    Returns:
        str: Cleaned text content or None if extraction failed
    """
    # Extract text from PDF
    raw_text = extract_text_from_pdf(pdf_bytes)
    
    if not raw_text:
        print("Failed to extract text from PDF")
        return None
        
    # Clean the extracted text
    cleaned_text = clean_text(raw_text)
    
    # Only return if we have meaningful text
    if len(cleaned_text) > 500:  # Require at least 500 characters
        return cleaned_text
    else:
        print(f"Too little text extracted from PDF ({len(cleaned_text)} characters)")
        return None