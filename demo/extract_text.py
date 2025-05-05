import re
import os
import csv
import PyPDF2
from pathlib import Path
import argparse
import tqdm
import fitz  # PyMuPDF - better PDF extraction

OUTPUT_LIMIT = 16000

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file using PyMuPDF (better extraction).
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
    """
    try:
        # Try PyMuPDF first (better extraction)
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            return text
        except:
            # Fall back to PyPDF2 if PyMuPDF fails
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ''
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:  # Check if page extraction was successful
                        text += page_text + '\n'
                return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
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
    if original_length <= OUTPUT_LIMIT:
        print("Text is already under 16k characters, keeping all content.")
        return text
    
    # Define patterns to selectively remove sections in order of priority
    section_patterns = [
        # 1. Contributors/authors at the beginning
        {
            'name': 'contributors',
            'pattern': r'^.*?(?:university|research|institute|lab|inc|corp|ltd|\@|\(|\))\s*\n',
            'flags': re.IGNORECASE | re.DOTALL | re.MULTILINE
        },
        # # 2. Abstract
        # {
        #     'name': 'abstract',
        #     'pattern': r'abstract\s*\n.*?(?=\n\s*\d+\.?\s+\w+|\n\s*[A-Z][a-z]+\s*\n|\n\s*[A-Z][A-Z\s]+\n)',
        #     'flags': re.IGNORECASE | re.DOTALL
        # },
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
            'replacement': ''
        }
    ]
    
    # Apply patterns one by one until text is under 30k
    for section in section_patterns:
        if len(text) <= OUTPUT_LIMIT:
            break
            
        print(f"Removing {section['name']} to reduce text length ({len(text)} characters)")
        
        replacement = section.get('replacement', '')
        text = re.sub(section['pattern'], replacement, text, flags=section['flags'])
        
        # Check if we actually removed content
        if len(text) == len(original_text):
            print(f"No {section['name']} found to remove")
    
    # If still over 30k after removing all sections, truncate
    if len(text) > OUTPUT_LIMIT:
        print(f"Still over 30k after removing sections, truncating ({len(text)} characters)")
        text = text[:OUTPUT_LIMIT]
    
    # If we've lost too much content, revert to truncated original
    if len(text) < 0.5 * min(original_length, OUTPUT_LIMIT):
        print(f"Cleaned text too short ({len(text)} chars), reverting to truncated original")
        text = original_text[:OUTPUT_LIMIT]
    
    # Remove extra whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = text.strip()
    
    # Replace multiple consecutive spaces with a single space
    text = re.sub(r' +', ' ', text)
    
    print(f"Final text length: {len(text)} characters")
    return text

def save_to_csv(pdf_path, text, csv_path):
    """
    Save extracted text to a CSV file with proper handling for multiline text.
    
    Args:
        pdf_path (str): Path to the PDF file
        text (str): Cleaned text to save
        csv_path (str): Path to the CSV file
    """
    file_exists = os.path.isfile(csv_path)
    
    # Prepare the data
    filename = os.path.basename(pdf_path)
    
    # Replace newlines with a special token for CSV storage
    # This prevents issues with CSV line breaks while preserving paragraph structure
    text_for_csv = text.replace('\n', '')
    
    with open("OUTPUT/" + csv_path, 'a', newline='', encoding='utf-8') as file:
        # Use csv.writer with proper quoting and escaping
        writer = csv.writer(
            file, 
            quoting=csv.QUOTE_ALL,
            escapechar='\\',
            doublequote=True,
            lineterminator='\n'
        )
        
        # Write header if file doesn't exist
        if not file_exists:
            writer.writerow(['filename', 'article'])
        
        # Write the row
        writer.writerow([filename, text_for_csv])

def process_pdf_directory(pdf_dir, csv_path):
    """
    Process all PDF files in a directory and save their cleaned text to CSV.
    
    Args:
        pdf_dir (str): Path to the directory containing PDF files
        csv_path (str): Path to the CSV file
    """
    # Get list of all PDF files in the directory
    pdf_files = [f for f in Path(pdf_dir).glob("**/*.pdf")]
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process.")
    
    # Process each PDF file with a progress bar
    for pdf_file in tqdm.tqdm(pdf_files):
        print(f"\nProcessing {pdf_file.name}")
        text = extract_text_from_pdf(pdf_file)
        if text:
            cleaned_text = clean_text(text)
            
            # Only save if we have meaningful text
            if len(cleaned_text) > 500:  # Require at least 500 characters
                save_to_csv(pdf_file, cleaned_text, csv_path)
                print(f"Saved {pdf_file.name} to CSV ({len(cleaned_text)} characters)")
            else:
                print(f"Too little text extracted from {pdf_file} ({len(cleaned_text)} characters)")
        else:
            print(f"Failed to process {pdf_file}")
    
    print(f"Processing complete. Results saved to {csv_path}")

def main():
    parser = argparse.ArgumentParser(description='Extract text from PDF files and save to CSV')
    parser.add_argument('pdf_dir', help='Directory containing PDF files to process')
    parser.add_argument('--output', '-o', default='articles.csv', help='Output CSV file (default: articles.csv)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print detailed processing information')
    
    args = parser.parse_args()
    
    # Ensure the PDF directory exists
    if not os.path.isdir(args.pdf_dir):
        print(f"Error: Directory '{args.pdf_dir}' does not exist.")
        return
    
    process_pdf_directory(args.pdf_dir, args.output)

if __name__ == "__main__":
    main()