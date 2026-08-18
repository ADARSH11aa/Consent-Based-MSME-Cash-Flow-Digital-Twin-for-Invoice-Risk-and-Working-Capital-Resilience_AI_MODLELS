# ocr_extraction.py
#
# This is Model 4 from our spec: OCR Extraction & Confidence Scoring.
#
# What this file does, in simple words:
#   1. We are given an invoice file - it could be a normal PDF, a scanned PDF, or a photo (jpg/png)
#   2. If it's a normal PDF, the text is already inside it, so we just read it directly.
#   3. If it's a scanned PDF or a photo, there is no real text - just an image. So we use
#      OCR (Optical Character Recognition) using a tool called Tesseract to "read" the image.
#   4. Once we have the text (from either method), we search for 5 fields we care about:
#      invoice number, invoice date, due date, total amount, and customer name.
#   5. Each field gets a confidence score (how sure we are that we read it correctly).
#      If confidence is too low, we mark it as needs_verification = True, so a human
#      can double check it before we trust it.

import re
import io
import pytesseract
from PIL import Image
from dateutil import parser as dateparser


# If a field's confidence is below this number (out of 100), we don't fully trust it
# and mark it for a human to check.
CONFIDENCE_THRESHOLD = 80.0

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# ============================================================
# STEP 1: Get lines of text out of the file (either directly, or using OCR)
# ============================================================

def get_lines_using_ocr(image):
    """
    Takes a picture (a PIL Image) and runs Tesseract OCR on it.

    Tesseract gives us back EVERY WORD separately, along with a confidence score
    for each word (0 to 100). We don't want individual words - we want full lines,
    like "Invoice No: INV-700123". So we group the words back into lines using the
    line/paragraph/block numbers that Tesseract already gives us.

    Returns a list of lines. Each line is a dictionary like:
        {"text": "Invoice No: INV-700123", "confidence": 93.3}
    """

    # image_to_data gives us a dictionary with lots of lists inside it - one entry
    # per word that Tesseract found. Example: data["text"][5] is the 5th word found.
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    # We will collect words into this dictionary, grouped by which line they belong to.
    # The key is (block_number, paragraph_number, line_number) - basically "which line".
    lines_grouped = {}

    number_of_words = len(data["text"])

    for i in range(number_of_words):
        word_text = data["text"][i].strip()
        word_confidence = float(data["conf"][i])

        # Tesseract sometimes returns empty words, or a confidence of -1 meaning
        # "I couldn't read anything here". We skip those.
        if word_text == "":
            continue
        if word_confidence < 0:
            continue

        line_key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])

        # If we haven't seen this line before, start a new empty entry for it
        if line_key not in lines_grouped:
            lines_grouped[line_key] = {"words": [], "confidences": []}

        lines_grouped[line_key]["words"].append(word_text)
        lines_grouped[line_key]["confidences"].append(word_confidence)

    # Now turn the grouped dictionary into a simple list of lines, in order
    result_lines = []
    for line_key in sorted(lines_grouped.keys()):
        words_in_line = lines_grouped[line_key]["words"]
        confidences_in_line = lines_grouped[line_key]["confidences"]

        full_line_text = " ".join(words_in_line)
        average_confidence = sum(confidences_in_line) / len(confidences_in_line)

        result_lines.append({"text": full_line_text, "confidence": average_confidence})

    return result_lines


def get_lines_from_real_pdf_text(pdf_text):
    """
    If the PDF already has real, selectable text (it was made digitally, not scanned),
    we don't need OCR at all - we can just read the text directly. This is much faster
    and much more accurate than OCR.

    Since this text is exact (not a guess), we give every line a very high confidence
    score of 99, instead of running any calculation.
    """
    result_lines = []

    for raw_line in pdf_text.split("\n"):
        cleaned_line = raw_line.strip()
        if cleaned_line != "":
            result_lines.append({"text": cleaned_line, "confidence": 99.0})

    return result_lines


def read_invoice_file(file_bytes, filename):
    """
    This is the function that decides HOW to read the file, depending on its type.

    Returns two things:
        1. lines  -> a list of lines of text (see functions above)
        2. source -> the string "text_layer" if we read real PDF text,
                      or "ocr" if we had to use Tesseract to guess the text
    """

    file_extension = filename.lower().split(".")[-1]

    if file_extension == "pdf":
        # PyMuPDF (imported as "fitz") lets us open PDFs and also lets us turn a PDF
        # page into a picture if needed - so we don't need any extra tools like Poppler.
        import fitz

        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
        first_page = pdf_document[0]  # for now we only look at page 1 of the invoice

        pdf_text = first_page.get_text().strip()

        if len(pdf_text) > 20:
            # There's a good amount of real text here - this PDF was made digitally.
            lines = get_lines_from_real_pdf_text(pdf_text)
            source = "text_layer"
        else:
            # Almost no real text was found - this PDF is probably just a scanned image.
            # So we convert the page into a picture, then run OCR on that picture.
            pixmap = first_page.get_pixmap(dpi=300)
            page_image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            lines = get_lines_using_ocr(page_image)
            source = "ocr"

    else:
        # It's a normal image file (.jpg, .png, etc) - like a photo of a paper invoice.
        image = Image.open(io.BytesIO(file_bytes))
        lines = get_lines_using_ocr(image)
        source = "ocr"

    return lines, source


# ============================================================
# STEP 2: Search through the lines of text to find the fields we care about
# ============================================================

# For each field, we list the different "labels" that usually appear right before the
# value in an invoice. For example, an invoice number is usually written after something
# like "Invoice No:" or "Inv #". We check each of these possible labels.
FIELD_LABELS = {
    "invoice_number": [r"invoice\s*(no\.?|number|#)\s*:?", r"inv\s*#\s*:?"],
    "invoice_date": [r"invoice\s*date\s*:?", r"date\s*of\s*issue\s*:?"],
    "due_date": [r"due\s*date\s*:?", r"payment\s*due\s*:?"],
    "total_amount": [r"grand\s*total\s*:?", r"total\s*amount\s*:?", r"amount\s*due\s*:?"],
    "customer_name": [r"bill\s*to\s*:?", r"billed\s*to\s*:?", r"customer\s*name\s*:?"],
}

# For each field, this is the pattern the ACTUAL VALUE should match, once we've found
# the label. For example, once we find "Invoice No:", we expect the value right after it
# to look like letters/numbers/dashes, e.g. "INV-700123".
VALUE_PATTERNS = {
    "invoice_number": re.compile(r"[A-Z0-9][A-Z0-9\-\/]{3,}"),
    "invoice_date": re.compile(r"\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}"),
    "due_date": re.compile(r"\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}"),
    "total_amount": re.compile(r"(?:₹|rs\.?|inr)?\s*([\d,]+\.\d{1,2}|[\d,]+)", re.IGNORECASE),
    "customer_name": re.compile(r"[A-Za-z][A-Za-z0-9 &.,'\-]{2,}"),
}


def find_value_in_line(line_text, label_pattern, value_pattern):
    """
    Looks for a label (like "Invoice No:") somewhere in this one line of text.
    If we find the label, we look at whatever text comes AFTER it, and try to match
    the value pattern there (e.g. find something that looks like an invoice number).

    Returns the matched value as a string, or None if nothing was found.
    """

    label_match = re.search(label_pattern, line_text, re.IGNORECASE)

    if label_match is None:
        return None  # this line doesn't contain this label at all

    # take only the part of the line AFTER the label
    text_after_label = line_text[label_match.end():]

    value_match = value_pattern.search(text_after_label)

    if value_match is None:
        return None  # we found the label, but nothing after it looked like a valid value

    matched_text = value_match.group(0)
    matched_text = matched_text.strip(" :.-")  # remove leftover punctuation/spaces

    return matched_text


def clean_up_value(field_name, raw_text_value):
    """
    Takes the raw text we matched (e.g. "12/03/2026" or "Rs. 45,000.00") and converts it
    into a proper, clean value:
        - dates become "YYYY-MM-DD" text (easy to compare/sort)
        - amounts become actual numbers (float), with commas and currency symbols removed

    If something can't be converted properly (e.g. a date that doesn't make sense),
    we return None so the caller knows this needs a human to check it.
    """

    try:
        if field_name == "total_amount":
            no_commas = raw_text_value.replace(",", "")
            no_currency_symbols = no_commas.replace("₹", "").replace("Rs.", "").replace("INR", "")
            return float(no_currency_symbols.strip())

        if field_name == "invoice_date" or field_name == "due_date":
            parsed_date = dateparser.parse(raw_text_value, dayfirst=True)
            return parsed_date.date().isoformat()

    except Exception:
        return None

    # for fields like invoice_number and customer_name, no conversion needed
    return raw_text_value


def extract_fields_from_lines(lines, source):
    """
    Goes through every field we care about (invoice_number, invoice_date, due_date,
    total_amount, customer_name) and tries to find it inside the lines of text we have.

    Returns a dictionary where each field has its value, confidence score, and whether
    it needs a human to double-check it.
    """

    all_fields = {}

    for field_name in FIELD_LABELS:
        possible_labels = FIELD_LABELS[field_name]
        value_pattern = VALUE_PATTERNS[field_name]

        found_value = None
        found_confidence = None

        # go through every line of the invoice, one at a time
        for line in lines:
            # for this line, try every possible label for the current field
            for label_pattern in possible_labels:
                value = find_value_in_line(line["text"], label_pattern, value_pattern)

                if value is not None:
                    found_value = value
                    found_confidence = line["confidence"]
                    break  # stop trying other labels, we already found it on this line

            if found_value is not None:
                break  # stop checking other lines too, we already found the field

        if found_value is None:
            # we searched every line and never found this field - flag it for review
            all_fields[field_name] = {
                "value": None,
                "raw_value": None,
                "confidence": 0.0,
                "source": source,
                "needs_verification": True,
            }
            continue

        cleaned_value = clean_up_value(field_name, found_value)

        if cleaned_value is None:
            # we found something, but couldn't turn it into a proper date/number
            # treat this as low confidence even if the OCR itself was confident
            final_confidence = min(found_confidence, 50.0)
        else:
            final_confidence = found_confidence

        all_fields[field_name] = {
            "value": cleaned_value,
            "raw_value": found_value,
            "confidence": round(final_confidence, 1),
            "source": source,
            "needs_verification": final_confidence < CONFIDENCE_THRESHOLD,
        }

    return all_fields


# ============================================================
# This is the ONE function other files (like main.py) should call.
# ============================================================

def extract_invoice(file_bytes, filename):
    """
    Main entry point of this file.

    Give it:
        file_bytes - the raw bytes of the uploaded invoice file
        filename   - the original filename (used to check if it's a .pdf, .jpg, etc)

    Returns a dictionary with everything we extracted from the invoice.
    """

    lines, source = read_invoice_file(file_bytes, filename)
    fields = extract_fields_from_lines(lines, source)

    # if even ONE field needs a human to check it, we mark the whole invoice as needing review
    invoice_needs_review = False
    for field_name in fields:
        if fields[field_name]["needs_verification"] is True:
            invoice_needs_review = True

    return {
        "filename": filename,
        "source": source,
        "fields": fields,
        "needs_review": invoice_needs_review,
    }