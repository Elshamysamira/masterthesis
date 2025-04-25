import fitz

def pdf_to_txt(pdf_path, txt_path):
    doc = fitz.open(pdf_path)
    with open(txt_path, 'w', encoding='utf-8') as f:
        for page in doc:
            text = page.get_text()
            f.write(text)
        print(f"saved text to {txt_path}")

pdf_to_txt("questions_german/questions_german.pdf", "questions_german/questions_german.txt")