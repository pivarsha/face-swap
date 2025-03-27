import easyocr
import csv
import re

image_path = "image.jpg"
csv_path = "structured_extracted_data.csv"

def extract_phone_numbers(text):
    """Extract phone numbers from text."""
    phone_pattern = r'\(?\d{2,4}\)?[-.\s]?\d{3}[-.\s]?\d{4,5}'
    return re.findall(phone_pattern, text)

def extract_credit_card_info(text):
    """Extract credit card numbers from text."""
    cc_pattern = r'\b(?:\d[ -]*?){13,16}\b'
    return re.findall(cc_pattern, text)

def extract_alphanumeric_codes(text):
    """Extract alphanumeric codes (Serial numbers, IDs)."""
    alpha_pattern = r'\b[A-Za-z0-9@#/-]{8,}\b'
    return re.findall(alpha_pattern, text)

def extract_dates(text):
    """Extract dates in various formats."""
    date_pattern = r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}[ ]?(?:January|February|March|April|May|June|July|August|September|October|November|December)[ ]?\d{4})\b'
    return re.findall(date_pattern, text)

def extract_names(text):
    """Extract names (Assuming name appears before a date)."""
    name_pattern = r'\b(?:MR\.|MRS\.|MS\.|DR\.)\s+[A-Z][a-z]+\s+[A-Z]?\s*[A-Z][a-z]+\b'
    matches = re.findall(name_pattern, text)
    structured_names = []
    for name in matches:
        parts = name.split()
        first = parts[1] if len(parts) > 1 else ""
        middle = parts[2] if len(parts) > 2 else ""
        last = parts[-1] if len(parts) > 1 else ""
        structured_names.append((first, middle, last))
    return structured_names

try:
    reader = easyocr.Reader(["en"])

    extracted_text_list = reader.readtext(image_path, detail=0)  
    extracted_text = " ".join(extracted_text_list)

    names = extract_names(extracted_text)
    phone_numbers = extract_phone_numbers(extracted_text)
    dates = extract_dates(extracted_text)
    alphanumeric_codes = extract_alphanumeric_codes(extracted_text)
    credit_cards = extract_credit_card_info(extracted_text)

    misc_data = extracted_text
    address = "Unknown" 

    # Save structured data to CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["First Name", "Middle Name", "Last Name", "Phone Number", "Address", "Date", "Alphanumeric Codes", "Credit Card Info", "Miscellaneous"])
        
        for i in range(max(len(names), len(phone_numbers), len(dates), len(alphanumeric_codes), len(credit_cards))):
            row = [
                names[i][0] if i < len(names) else "",
                names[i][1] if i < len(names) else "",
                names[i][2] if i < len(names) else "",
                phone_numbers[i] if i < len(phone_numbers) else "",
                address,
                dates[i] if i < len(dates) else "",
                alphanumeric_codes[i] if i < len(alphanumeric_codes) else "",
                credit_cards[i] if i < len(credit_cards) else "",
                misc_data
            ]
            writer.writerow(row)

    print(f"Structured text successfully saved to {csv_path}")

except FileNotFoundError:
    print(f"Error: The image file '{image_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
