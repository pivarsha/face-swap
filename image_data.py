# import pytesseract
# from PIL import Image
# import re
# import csv

# # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# def process_extracted_text(text):
#     # Enhanced regex patterns
#     patterns = {
#         'Names and Titles': re.compile(r'(?i)(Mr|Ms|Mrs|Dr)\.?\s+([A-Z][a-z]+(?:\s+[A-Z])?(?:\s+[A-Z][a-z]+)+)'),
#         'Phone Numbers': re.compile(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'),
#         'Dates': re.compile(r'(\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b)|(\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+\w+\s+\d{1,2},\s+\d{4}\b)|(\d{1,2}-[A-Za-z]{3,4}-\d{2,4})', re.IGNORECASE),
#         'Addresses': re.compile(r'\d+\s.+?(?=\s\b(?:[A-Z]{2}|Austria|Ontario)\b)[^,\d]+?(?:\b(?:[A-Z]{2}|Austria|Ontario)\b\s*\d{5})?', re.IGNORECASE),
#         'Alphanumeric Codes': re.compile(r'\b(?:[A-Za-z0-9@$#/-]{10,}|[A-Za-z]{3,}\d{5,})\b'),
#         'Credit Card Information': re.compile(r'\b(?:American\s+Express|VISA\s+Gold|Discover\s+Gold)\b.*?\d{12,19}\b', re.IGNORECASE),
#         'Miscellaneous': re.compile(r'\b(?:platz|FIRM CASES|LOWER AUSTRIA|old ripley|Ottawa Massachusetts|Des Moines|north bay ontario)\b.*?\d{5}', re.IGNORECASE)
#     }

#     results = {}
#     for category, pattern in patterns.items():
#         matches = pattern.findall(text)
#         # Flatten matches and clean results
#         cleaned_matches = [' '.join(m) if isinstance(m, tuple) else m for m in matches]
#         results[category] = [m.strip() for m in cleaned_matches if m.strip()]
    
#     return results

# def save_to_csv(data, filename):
#     # Determine the maximum length of the lists
#     max_length = max(len(items) for items in data.values())
    
#     # Write to CSV
#     with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
#         writer = csv.writer(csvfile)
        
#         # Write header
#         headers = ["Names and Titles", "Phone Numbers", "Dates", "Addresses", "Alphanumeric Codes", "Credit Card Information", "Miscellaneous"]
#         writer.writerow(headers)
        
#         # Write data row-wise
#         for i in range(max_length):
#             row = [
#                 data['Names and Titles'][i] if i < len(data['Names and Titles']) else '',
#                 data['Phone Numbers'][i] if i < len(data['Phone Numbers']) else '',
#                 data['Dates'][i] if i < len(data['Dates']) else '',
#                 data['Addresses'][i] if i < len(data['Addresses']) else '',
#                 data['Alphanumeric Codes'][i] if i < len(data['Alphanumeric Codes']) else '',
#                 data['Credit Card Information'][i] if i < len(data['Credit Card Information']) else '',
#                 data['Miscellaneous'][i] if i < len(data['Miscellaneous']) else ''
#             ]
#             writer.writerow(row)

# # Main processing
# image_path = "image.jpg"
# img = Image.open(image_path)
# extracted_text = pytesseract.image_to_string(img)

# processed_data = process_extracted_text(extracted_text)
# save_to_csv(processed_data, "structured_data.csv")

# print("Structured data saved to structured_data.csv")


import pytesseract
from PIL import Image
import pandas as pd
import csv

# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows example

image_path = "image.jpg"
img = Image.open(image_path)

extracted_text = pytesseract.image_to_string(img)

csv_path = "extracted_text.csv"
with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Extracted Text"])  # Header
    writer.writerow([extracted_text])    # Text data

print(f"Text saved to {csv_path}")