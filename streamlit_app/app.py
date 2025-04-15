import streamlit as st
from pdf_extraction import process_pdf
# from convert import generate_answers

def main():
    # Set the app title
    st.title("PDF Text Extractor")
    
    # Add description
    st.write("Upload a PDF file to extract its text content.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Display file details
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.2f} KB"
        }
        st.write("File Details:")
        for key, value in file_details.items():
            st.write(f"- {key}: {value}")
        
        # Extract text on button click
        if st.button("Extract pdf info"):
            with st.spinner("Extracting text..."):
                # Extract text from the PDF
                extracted_text = process_pdf(uploaded_file)
                # summary = generate_answers(extracted_text)


                # print("TYPE OF SUMMARY", type(summary))
                # print("LEN OF SUMMARY", len(summary))

                
                st.subheader("Extracted Text:")
                st.text_area("", extracted_text, height=200)

if __name__ == "__main__":
    main()
