"""
extractor.py
------------
Field extractor for Invoice, Resume, and Utility Bill documents.

FIXES APPLIED:
  1. extract_amount_due   — Added $ £ € to currency patterns (was PKR/Rs only)
                            Added explicit "PAYABLE WITHIN DUE DATE" pattern
  2. extract_usage_kwh    — Reordered patterns: specific first, generic last
                            Generic '[digits]+kwh' pattern was matching meter readings
                            (e.g. 14320 kWh) before reaching "units consumed"
  3. extract_invoice_number — Changed [A-Z0-9] to [A-Za-z0-9] to catch
                            mixed-case IDs like inv-001, INV-2024-001
  4. extract_experience_years — Fallback now calculates actual year delta
                            instead of returning len(job_ranges) which was just
                            a count of jobs. Also added unicode en/em dash
                            support (U+2013, U+2014) which PDFs commonly use.
"""

import re
import datetime
from typing import Optional


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def remove_spaced_text(text: str) -> str:
    """
    Fix PDFs where text is extracted as 'D E S C R I P T I O N'.
    Convert back to 'DESCRIPTION'.
    """
    fixed = re.sub(
        r'\b([A-Z])\s([A-Z])\s([A-Z])',
        lambda m: m.group(0).replace(' ', ''),
        text
    )
    return fixed


def clean_field(value: str) -> str:
    """Remove noise from extracted values."""
    if not value:
        return value
    value = re.sub(r'[\n\r\t]+', ' ', value)
    value = re.sub(r'\s+', ' ', value).strip()
    return value


# ─────────────────────────────────────────────
# INVOICE EXTRACTORS
# ─────────────────────────────────────────────

def extract_invoice_number(text: str) -> Optional[str]:
    """
    Extract invoice number.
    FIX: Changed char class from [A-Z0-9] → [A-Za-z0-9] so mixed-case
    IDs like 'inv-001' or 'Inv-2024-A' are captured correctly.
    """
    patterns = [
        r'invoice\s*[#:no\.]+\s*:?\s*([A-Za-z0-9][A-Za-z0-9\-/#]{2,20})',
        r'invoice\s*number\s*:?\s*([A-Za-z0-9][A-Za-z0-9\-/]{2,20})',
        r'invoice\s*no\.?\s*:?\s*([A-Za-z0-9][A-Za-z0-9\-/]{2,20})',
        r'\binv[-#\s]\s*([A-Za-z0-9][A-Za-z0-9\-/]{2,20})',
        r'bill\s*no\.?\s*:?\s*([A-Za-z0-9][A-Za-z0-9\-/]{2,20})',
        r'#\s*([A-Za-z0-9]{3,15})\b',
    ]

    blacklist = ['for', 'the', 'and', 'from', 'date', 'to', 'due', 'pay']

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result = match.group(1).strip()
            if result.lower() not in blacklist and len(result) >= 3:
                return result
    return None


def extract_date(text: str) -> Optional[str]:
    """Extract first meaningful date from document."""
    patterns = [
        r'\b(\d{4}[-/]\d{2}[-/]\d{2})\b',
        r'\b(\d{2}[-/]\d{2}[-/]\d{4})\b',
        r'\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\b',
        r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def extract_company(text: str) -> Optional[str]:
    """
    Extract company name.
    Look near 'from', 'company', 'vendor' labels first.
    Fall back to first clean line heuristic.
    """
    # Tuples of (pattern, use_ignorecase).
    # FIX: ALL-CAPS check must be case-SENSITIVE (False).
    # With re.IGNORECASE, [A-Z\s] also matches lowercase, so any
    # plain text like "Acme Corp" would falsely match ^[A-Z\s]+$ and
    # get flagged as noise — causing the function to always return None.
    blacklist_patterns = [
        (r'^[A-Z\s]+$',                                    False),  # ALL-CAPS headers only
        (r'D\s*E\s*S\s*C',                                 True),   # Spaced text artifacts
        (r'subtotal|discount|qty|unit|price|amount|total', True),   # Table column noise
    ]

    label_patterns = [
        r'(?:from|vendor|company|issued?\s*by|billed?\s*by)\s*:?\s*\n?\s*([A-Za-z0-9][^\n]{2,50})',
        r'(?:company\s*name)\s*:?\s*([A-Za-z0-9][^\n]{2,50})',
    ]

    for pattern in label_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result = clean_field(match.group(1))
            is_noise = any(
                re.search(bp, result, re.IGNORECASE if use_ic else 0)
                for bp, use_ic in blacklist_patterns
            )
            if not is_noise and len(result) > 2:
                return result

    # Fallback: first line that looks like a company name
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    for line in lines[:8]:
        if len(line) < 3 or len(line) > 60:
            continue
        if re.search(r'D\s*E\s*S|invoice|bill|date|#', line, re.IGNORECASE):
            continue
        if re.match(r'^[A-Za-z0-9][A-Za-z0-9\s&.,\-]+$', line):
            return line

    return None


def extract_total_amount(text: str) -> Optional[float]:
    """
    Extract total amount — prioritize 'total' near currency.
    Most specific patterns first to avoid matching subtotals or line items.
    """
    patterns = [
        r'(?:grand\s*)?total\s*amount\s*:?\s*[\$£€Rs₨]?\s*([\d,]+\.?\d*)',
        r'amount\s*due\s*:?\s*[\$£€Rs₨]?\s*([\d,]+\.?\d*)',
        r'net\s*payable\s*:?\s*[\$£€Rs₨]?\s*([\d,]+\.?\d*)',
        r'total\s*due\s*:?\s*[\$£€Rs₨]?\s*([\d,]+\.?\d*)',
        r'\btotal\b\s*:?\s*[\$£€Rs₨]?\s*([\d,]+\.?\d{2})\b',
        r'[\$£€]\s*([\d,]+\.\d{2})\s*$',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            try:
                return float(match.group(1).replace(',', ''))
            except ValueError:
                continue
    return None


# ─────────────────────────────────────────────
# RESUME EXTRACTORS
# ─────────────────────────────────────────────

def extract_name(text: str) -> Optional[str]:
    """Extract name — label-based first, then first-line heuristic."""
    patterns = [
        r'(?:^|\n)\s*name\s*:?\s*([A-Za-z][A-Za-z\s]{2,40})',
        r'applicant\s*:?\s*([A-Za-z][A-Za-z\s]{2,40})',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # First line heuristic — 2 to 4 words, letters only
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    for line in lines[:5]:
        if re.match(r'^[A-Za-z]+(?:\s[A-Za-z]+){1,3}$', line):
            if len(line) > 4:
                return line
    return None


def extract_email(text: str) -> Optional[str]:
    match = re.search(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', text)
    return match.group(0) if match else None


def extract_phone(text: str) -> Optional[str]:
    patterns = [
        r'(\+92[\s\-]?\d{3}[\s\-]?\d{7})',
        r'(0\d{3}[\s\-]?\d{7})',
        r'(\+\d{1,3}[\s\-]\d{3}[\s\-]\d{7,8})',
        r'(\(?\d{3}\)?[\s\-]\d{3}[\s\-]\d{4})',
        r'(\d{10,13})',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return None


def extract_experience_years(text: str) -> Optional[int]:
    """
    Extract years of experience.

    FIX 1: Fallback now calculates actual total years from date ranges
            instead of returning len(job_ranges) (which just counted jobs).
    FIX 2: Unicode en-dash \u2013 and em-dash \u2014 added — PDFs often
            store '2020–2023' with these instead of a plain hyphen.
    """
    # Direct mention patterns first
    patterns = [
        r'(\d+)\+?\s*years?\s*of\s*(?:professional\s*|relevant\s*|work\s*)?experience',
        r'experience\s*(?:of\s*)?(\d+)\+?\s*years?',
        r'(\d+)\+?\s*years?\s*(?:in\s*)?(?:the\s*)?(?:industry|field|domain)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))

    # Fallback: calculate actual years from job date ranges
    # FIX: use \u2013 (en-dash) and \u2014 (em-dash) in addition to hyphen
    job_ranges = re.findall(
        r'\b(20\d{2}|19\d{2})\s*[-\u2013\u2014to]+\s*(20\d{2}|19\d{2}|present|current|now)',
        text, re.IGNORECASE
    )

    if job_ranges:
        current_year = datetime.datetime.now().year
        total_years = 0
        for start, end in job_ranges:
            s = int(start)
            e = current_year if end.lower() in ('present', 'current', 'now') else int(end)
            total_years += max(0, e - s)
        return total_years if total_years > 0 else None

    return None


# ─────────────────────────────────────────────
# UTILITY BILL EXTRACTORS
# ─────────────────────────────────────────────

def extract_account_number(text: str) -> Optional[str]:
    patterns = [
        r'(?:account|consumer|reference|customer)\s*(?:number|no|#|id)\s*:?\s*([A-Z0-9\-]{4,25})',
        r'(?:ref|acc|con)[\.\s#]*:?\s*([A-Z0-9\-]{4,25})',
        r'meter\s*(?:number|no)\s*:?\s*([A-Z0-9\-]{4,25})',
        r'(?:14\s*digit|consumer)\s*(?:reference)?\s*(?:no|number)\s*:?\s*(\d{10,16})',
        r'\b(\d{13,14})\b',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def extract_usage_kwh(text: str) -> Optional[float]:
    """
    Extract units consumed.

    FIX: Reordered patterns — specific labels first, bare 'kwh' last.
    Previously the bare kwh pattern ran first and matched the meter reading
    value (e.g. 14320 kWh) before the 'units consumed' pattern got a chance.
    """
    patterns = [
        # ── SPECIFIC first ─────────────────────────────────────────────────
        r'units?\s*consumed\s*:?\s*([\d,]+\.?\d*)',          # "Units Consumed: 227"
        r'(?:billed|charged)\s*units?\s*:?\s*([\d,]+)',
        r'consumption\s*:?\s*([\d,]+\.?\d*)',
        r'(?:current|net)\s*units?\s*:?\s*([\d,]+)',
        r'(?:cr|current\s*reading)\s*[-:]\s*(?:prev(?:ious)?\s*reading\s*[-:]?\s*\d+\s*)?'
        r'(?:units?\s*:?)?\s*([\d]+)',
        # ── GENERIC last — can match readings, so only as fallback ─────────
        r'([\d,]+\.?\d*)\s*kwh',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1).replace(',', ''))
            except ValueError:
                continue
    return None


def extract_amount_due(text: str) -> Optional[float]:
    """
    Extract amount due from utility bills.

    FIX: Added $, £, € to all currency patterns — previous version only
    checked for Rs./PKR/₨ so bills priced in dollars returned null.
    Added explicit 'PAYABLE WITHIN DUE DATE' pattern to catch that exact
    label used on the generated sample bills.
    """
    patterns = [
        # Explicit payable-within-due-date label (matches generated bills)
        r'payable\s*within\s*due\s*date\s*[\$£€Rs\.PKR₨]?\s*([\d,]+\.?\d*)',
        # Standard payable patterns — now include $ £ €
        r'(?:net\s*)?payable\s*(?:amount)?\s*:?\s*[\$£€Rs\.PKR₨]?\s*([\d,]+\.?\d*)',
        r'amount\s*(?:payable|due)\s*:?\s*[\$£€Rs\.PKR₨]?\s*([\d,]+\.?\d*)',
        r'total\s*(?:amount\s*)?due\s*:?\s*[\$£€Rs\.PKR₨]?\s*([\d,]+\.?\d*)',
        r'please\s*pay\s*:?\s*[\$£€Rs\.PKR₨]?\s*([\d,]+\.?\d*)',
        r'(?:within\s*due\s*date|after\s*due\s*date)\s*:?\s*[\$£€Rs\.PKR₨]?\s*([\d,]+\.?\d*)',
        # Fallback — any amount preceded by a known currency symbol
        r'[\$£€]\s*([\d,]+\.?\d*)',
        r'(?:Rs\.?|PKR|₨)\s*([\d,]+\.?\d*)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1).replace(',', ''))
            except ValueError:
                continue
    return None


# ─────────────────────────────────────────────
# MAIN DISPATCHER
# ─────────────────────────────────────────────

def extract_fields(text: str, doc_class: str) -> dict:
    if doc_class == "Invoice":
        return {
            "invoice_number": extract_invoice_number(text),
            "date":           extract_date(text),
            "company":        extract_company(text),
            "total_amount":   extract_total_amount(text),
        }
    elif doc_class == "Resume":
        return {
            "name":             extract_name(text),
            "email":            extract_email(text),
            "phone":            extract_phone(text),
            "experience_years": extract_experience_years(text),
        }
    elif doc_class == "Utility Bill":
        return {
            "account_number": extract_account_number(text),
            "date":           extract_date(text),
            "usage_kwh":      extract_usage_kwh(text),
            "amount_due":     extract_amount_due(text),
        }
    else:
        return {}