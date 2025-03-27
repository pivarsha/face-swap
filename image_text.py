
# import pytesseract
# from PIL import Image
# import re
# import csv

# # Set the path to the Tesseract executable (if not in PATH)
# # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows example

# # Open the image file
# image_path = "1000080048.jpg"  # Replace with your image path
# img = Image.open(image_path)

# # Extract text from the image
# extracted_text = pytesseract.image_to_string(img)

# # Function to extract structured data
# def extract_structured_data(text):
#     # Regular expressions for matching
#     email_regex = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'  # Matches emails
#     phone_regex = r'\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}'  # Matches phone numbers
#     date_regex = r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b'  # Matches dates
#     name_regex = r'(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s+([A-Za-z]+)\s+([A-Za-z]+)\s+([A-Za-z]+)'  # Matches names with titles and three parts
#     address_regex = r'\d+\s+[A-Za-z]+\s+[A-Za-z]+'  # Matches simple addresses (e.g., 123 Main Street)

#     # Extract data using regex
#     emails = re.findall(email_regex, text)
#     phone_numbers = re.findall(phone_regex, text)
#     dates = re.findall(date_regex, text)
#     names = re.findall(name_regex, text)
#     addresses = re.findall(address_regex, text)

#     # Combine data into rows
#     structured_data = []
#     for i in range(max(len(emails), len(phone_numbers), len(dates), len(names), len(addresses))):
#         # Split name into first, middle, and last name
#         if i < len(names):
#             first_name, middle_name, last_name = names[i]
#         else:
#             first_name, middle_name, last_name = "", "", ""

#         row = {
#             "Email": emails[i] if i < len(emails) else "",
#             "First Name": first_name,
#             "Middle Name": middle_name,
#             "Last Name": last_name,
#             "Phone Number": phone_numbers[i] if i < len(phone_numbers) else "",
#             "Address": addresses[i] if i < len(addresses) else "",
#             "Date": dates[i] if i < len(dates) else ""
#         }
#         structured_data.append(row)

#     return structured_data

# # Extract structured data
# structured_data = extract_structured_data(extracted_text)

# # Save the structured data to a CSV file
# csv_path = "structured_data.csv"
# with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
#     fieldnames = ["Email", "First Name", "Middle Name", "Last Name", "Phone Number", "Address", "Date"]
#     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
#     writer.writeheader()  # Write header
#     for row in structured_data:
#         writer.writerow(row)

# print(f"Structured data saved to {csv_path}")




import pytesseract
from PIL import Image
import pandas as pd
import csv

# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows example

image_path = "1000080048.jpg"
img = Image.open(image_path)

extracted_text = pytesseract.image_to_string(img)

csv_path = "extracted_text.csv"
with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Extracted Text"])  # Header
    writer.writerow([extracted_text])    # Text data

print(f"Text saved to {csv_path}")




# import pytesseract
# from PIL import Image
# import pandas as pd
# import re

# # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows example

# # Load the image and extract text
# image_path = "1000080048.jpg"
# img = Image.open(image_path)
# extracted_text = pytesseract.image_to_string(img)

# # Define regex patterns for extraction
# email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
# phone_pattern = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
# date_pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}[th|st|nd|rd]?\s(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
# name_pattern = r'(?:Mr\.|Ms\.|Mrs\.|Dr\.)\s[A-Za-z]+\s[A-Za-z]*\s?[A-Za-z]+'
# address_pattern = r'\d+\s[\w\s]+,\s[\w\s]+,\s[A-Z]{2}\s\d{5}'  # Example: 123 Main St, Springfield, IL 62704

# # Extract data using regex
# emails = re.findall(email_pattern, extracted_text)
# phones = re.findall(phone_pattern, extracted_text)
# dates = re.findall(date_pattern, extracted_text)
# names = re.findall(name_pattern, extracted_text)
# addresses = re.findall(address_pattern, extracted_text)

# # Create a DataFrame to store the structured data
# data = []
# for i in range(len(names)):
#     row = {
#         "First Name": names[i].split()[1] if len(names[i].split()) > 1 else "",
#         "Middle Name": names[i].split()[2] if len(names[i].split()) > 2 else "",
#         "Last Name": names[i].split()[3] if len(names[i].split()) > 3 else "",
#         "Email": emails[i] if i < len(emails) else "",
#         "Phone Number": phones[i] if i < len(phones) else "",
#         "Address": addresses[i] if i < len(addresses) else "",
#         "Date": dates[i] if i < len(dates) else "",
#     }
#     data.append(row)

# # Convert to DataFrame
# df = pd.DataFrame(data)

# # Save to Excel
# excel_path = "structured_data.xlsx"
# df.to_excel(excel_path, index=False)

# print(f"Structured data saved to {excel_path}")

import pytesseract
from PIL import Image
import pandas as pd
import re

image_path = "1000080048.jpg"
img = Image.open(image_path)
extracted_text = pytesseract.image_to_string(img)

txt_path = "extracted_text.txt"
with open(txt_path, 'w', encoding='utf-8') as txt_file:
    txt_file.write(extracted_text)

print(f"Text extracted and saved to {txt_path}")

def process_extracted_text(text):
    name_pattern = re.compile(r'(Mr\.|Ms\.|Mrs\.|Dr\.)\s+([A-Za-z]+)\s+([A-Za-z]*)\s*([A-Za-z]+)')
    email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
    phone_pattern = re.compile(r'\(\d{3}\)\s*\d{3}-\d{4}')
    date_pattern = re.compile(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}')
    alphanumeric_pattern = re.compile(r'\b[A-Za-z0-9]{6,}\b')
    address_pattern = re.compile(r'\d+\s+[A-Za-z]+\s+[A-Za-z]+')

    names = name_pattern.findall(text)
    emails = email_pattern.findall(text)
    phones = phone_pattern.findall(text)
    dates = date_pattern.findall(text)
    alphanumerics = alphanumeric_pattern.findall(text)
    addresses = address_pattern.findall(text)

    max_length = max(
        len(names), len(emails), len(phones), len(dates), len(alphanumerics), len(addresses)
    )

    names += [('', '', '', '')] * (max_length - len(names))
    emails += [''] * (max_length - len(emails))
    phones += [''] * (max_length - len(phones))
    dates += [''] * (max_length - len(dates))
    alphanumerics += [''] * (max_length - len(alphanumerics))
    addresses += [''] * (max_length - len(addresses))

    data = {
        'Title': [n[0] for n in names],
        'First Name': [n[1] for n in names],
        'Middle Name': [n[2] for n in names],
        'Last Name': [n[3] for n in names],
        'Email': emails,
        'Phone Number': phones,
        'Date': dates,
        'Alphanumeric Codes': alphanumerics,
        'Address': addresses
    }

    df = pd.DataFrame(data)
    return df

processed_data = process_extracted_text(extracted_text)

csv_path = "processed_data.csv"
processed_data.to_csv(csv_path, index=False)

print(f"Processed data saved to {csv_path}")
