"""
classifier.py
-------------
Classifies documents using keyword-based scoring.

WHY THIS APPROACH:
- No training data required
- Fast and interpretable
- Keywords are domain-specific and reliable for these document types
- TF-IDF style scoring gives weighted confidence per category

FIXES:
- Raised minimum score threshold from 3 → 8 to prevent research papers /
  generic docs from being misclassified as Resume (they score low but > 3)
- "Other" fallback now correctly catches ambiguous low-confidence results
"""

import re
from typing import Literal

# Define keyword sets for each category
# More specific / longer keywords = higher weight score
KEYWORDS = {
    "Invoice": [
        "invoice", "inv-", "bill to", "ship to", "subtotal", "total amount",
        "tax", "vat", "gst", "payment due", "due date", "invoice number",
        "invoice date", "item", "quantity", "unit price", "amount due",
        "vendor", "purchase order", "po number", "billing"
    ],
    "Resume": [
        "resume", "curriculum vitae", "cv", "objective", "summary",
        "experience", "education", "skills", "employment", "work history",
        "university", "bachelor", "master", "degree", "gpa", "internship",
        "references", "linkedin", "github", "certification", "projects",
        "achievements", "languages", "proficient", "years of experience"
    ],
    "Utility Bill": [
        "utility", "electricity", "kwh", "kilowatt", "meter reading",
        "account number", "service address", "billing period", "usage",
        "gas", "water", "units consumed", "tariff", "meter number",
        "current reading", "previous reading", "consumption", "supply"
    ]
}


def classify_document(
    text: str,
) -> Literal["Invoice", "Resume", "Utility Bill", "Other", "Unclassifiable"]:
    """
    Classify document text into one of 5 categories.

    Method: Count keyword matches per category.
    Category with highest score wins.

    Thresholds:
      score == 0          → Unclassifiable  (no signal at all)
      score < 8           → Other           (FIX: was 3, too low — research
                                             papers scored 4 and passed through)
      top/second > 0.85   → Other           (too ambiguous to decide)
      else                → top category
    """
    if not text or len(text.strip()) < 20:
        return "Unclassifiable"

    text_lower = text.lower()
    scores = {}

    for category, keywords in KEYWORDS.items():
        score = 0
        for keyword in keywords:
            count = text_lower.count(keyword.lower())
            if count > 0:
                # Longer / more specific keywords score higher
                weight = len(keyword.split()) + 1
                score += count * weight
        scores[category] = score

    # Sort by score descending
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_category, top_score = sorted_scores[0]
    second_score = sorted_scores[1][1]

    # ── Thresholds ──────────────────────────────────────────────────────────
    if top_score == 0:
        return "Unclassifiable"

    # FIX: raised from 3 → 8
    # Real invoices/resumes/bills score 15-60+; ambiguous docs score 2-7
    elif top_score < 8:
        return "Other"

    elif second_score > 0 and (second_score / top_score) > 0.85:
        # Scores too close — not confident enough
        return "Other"

    else:
        return top_category


def get_confidence_scores(text: str) -> dict:
    """Return raw scores for all categories — useful for debugging."""
    text_lower = text.lower()
    scores = {}
    for category, keywords in KEYWORDS.items():
        score = sum(
            text_lower.count(kw.lower()) * (len(kw.split()) + 1)
            for kw in keywords
        )
        scores[category] = score
    return scores