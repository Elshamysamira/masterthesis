import fitz
import os

def pdf_to_txt_batch(input_folder, output_folder):
    # Make sure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all PDF files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(input_folder, filename)
            txt_filename = filename.replace('.pdf', '.txt')
            txt_path = os.path.join(output_folder, txt_filename)

            # Convert each PDF to a TXT file
            doc = fitz.open(pdf_path)
            with open(txt_path, 'w', encoding='utf-8') as f:
                for page in doc:
                    text = page.get_text()
                    f.write(text)
            print(f"Saved text to {txt_path}")

# Usage
pdf_to_txt_batch("documents/pdf_german", "documents/txt_german")
pdf_to_txt_batch("documents/pdf_english", "documents/txt_english")