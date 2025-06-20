import fitz

def pdf_to_txt(pdf_path, txt_path):
    doc = fitz.open(pdf_path)
    with open(txt_path, 'w', encoding='utf-8') as f:
        for page in doc:
            text = page.get_text()
            f.write(text)
        print(f"saved text to {txt_path}")

pdf_to_txt("queries/queries_english/questions_english.pdf", "queries/queries_english/questions_english.txt")