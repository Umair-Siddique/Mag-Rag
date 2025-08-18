import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import docx2txt
import re

# --- Configuration for Windows ---
files_folder_path = "dataset\dataset_go_to_market"
output_folder_path = "extracted_data\extracted_text_go_to_market"
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- Setup ---
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)


# --- NEW: Filename Sanitizer ---
def sanitize_filename(filename):
    """
    Cleans a filename to be ASCII-safe for use in vector IDs.
    """
    # Separate the name from the extension
    name, ext = os.path.splitext(filename)
    
    # Encode to ASCII, ignoring any characters that can't be converted
    sanitized_name = name.encode('ascii', 'ignore').decode('ascii')
    
    # Replace spaces with underscores
    sanitized_name = sanitized_name.replace(' ', '_')
    
    # Remove any remaining characters that aren't letters, numbers, underscores, or hyphens
    sanitized_name = re.sub(r'[^\w-]', '', sanitized_name)
    
    # Return the sanitized name with a .txt extension
    return f"{sanitized_name}.txt"


# --- Preprocessing Method (Unchanged) ---
def preprocess_text(text):
    """
    Cleans extracted text for a RAG system by removing URLs, references, and OCR artifacts.
    """
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Remove references/citations like [1], [2,3], etc.
    text = re.sub(r'\[\d+(,\s*\d+)*\]', '', text)

    # Remove standalone numeric lines (often page numbers)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # Remove sequences of dots, underscores, or hyphens longer than 3
    text = re.sub(r'(\.\s*){3,}', ' ', text)
    text = re.sub(r'_{3,}', ' ', text)
    text = re.sub(r'-{3,}', ' ', text)

    # Replace ligatures and special quotes with ASCII characters
    replacements = {
        'ﬁ': 'fi',
        'ﬂ': 'fl',
        'ﬀ': 'ff',
        '“': '"', '”': '"',
        '‘': "'", '’': "'"
    }
    for search, replace in replacements.items():
        text = text.replace(search, replace)

    # Remove unwanted special characters except punctuation
    text = re.sub(r'[^\w\s\.,\'"\!\?\-\(\)\[\]]', '', text)

    # Normalize whitespace and tabs to single spaces
    text = re.sub(r'[ \t]+', ' ', text)

    # Normalize multiple newlines (more than 2) to just 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove isolated short numeric or special-character lines (common OCR issues)
    text = re.sub(r'^\W{1,3}\s*$', '', text, flags=re.MULTILINE)

    # Strip lines and remove empty lines
    lines = (line.strip() for line in text.split('\n'))
    non_empty_lines = [line for line in lines if line.strip()]

    return '\n'.join(non_empty_lines)



# --- The Pipeline (Modified) ---
def process_all_files(source_directory, output_directory):
    """
    Extracts and cleans text from all PDF and Word files in a directory.
    """
    print(f"Scanning for files in '{os.path.abspath(source_directory)}'...")
    
    for filename in os.listdir(source_directory):
        file_path = os.path.join(source_directory, filename)
        
        # --- MODIFIED: Use the sanitizer for the output filename ---
        sanitized_output_filename = sanitize_filename(filename)
        output_path = os.path.join(output_directory, sanitized_output_filename)
        # --- END MODIFICATION ---

        full_text = ""

        try:
            # --- Logic for PDF Files ---
            if filename.lower().endswith(".pdf"):
                is_image_based = False
                doc = fitz.open(file_path)
                
                for page in doc:
                    full_text += page.get_text()

                if len(full_text.strip()) < 100:
                    print(f"-> '{filename}' seems image-based, using OCR...")
                    is_image_based = True
                    full_text = ""
                    for page in doc:
                        pix = page.get_pixmap(dpi=200) 
                        img = Image.open(io.BytesIO(pix.tobytes()))
                        full_text += pytesseract.image_to_string(img)

                method = "OCR" if is_image_based else "Direct"
                print(f"✅ Success: Processed PDF '{filename}' (Method: {method})")

            # --- Logic for Word Files ---
            elif filename.lower().endswith((".docx", ".doc")):
                full_text = docx2txt.process(file_path)
                print(f"✅ Success: Processed Word file '{filename}'")

            # --- Clean and save the extracted text ---
            if full_text:
                cleaned_text = preprocess_text(full_text)
                
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(cleaned_text)

        except Exception as e:
            print(f"❌ Error processing '{filename}': {e}")


# --- Run the pipeline ---
if __name__ == "__main__":
    process_all_files(files_folder_path, output_folder_path)
    print("\nPipeline finished.")