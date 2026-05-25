import os
import json
import argparse
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError



from classifier import classify_document, get_confidence_scores
from extractor import extract_fields
from search import SemanticSearchEngine

import logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)


def extract_text_from_pdf(filepath: str) -> str:
    try:
        text = extract_text(filepath)
        return clean_text(text)
    except PDFSyntaxError:
        print(f"  [WARNING] Could not parse PDF: {filepath}")
        return ""
    except Exception as e:
        print(f"  [ERROR] {filepath}: {e}")
        return ""


def extract_text_from_txt(filepath: str) -> str:
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return clean_text(f.read())
    except Exception as e:
        print(f"  [ERROR] {filepath}: {e}")
        return ""


def clean_text(text: str) -> str:
    import re
    text = re.sub(r'[^\x20-\x7E\n\t]', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def filename_hint(filename: str) -> str:
    """
    Classify based on filename when text extraction fails or is weak.
    Pakistani bills (MEPCO/LESCO) are often image-based PDFs — 
    pdfminer extracts very little text from them, so filename is key.
    """
    name = filename.lower()

    # Utility bill hints
    if any(w in name for w in [
        'bill', 'utility', 'electric', 'gas', 'water',
        'lesco', 'wapda', 'mepco', 'iesco', 'fesco',
        'gepco', 'hesco', 'sngpl', 'sui',
        'bill_pak', 'electricity', 'bijli'
    ]):
        return "Utility Bill"

    # Invoice hints
    if any(w in name for w in ['invoice', 'inv-', 'receipt', 'purchase']):
        return "Invoice"

    # Resume hints
    if any(w in name for w in ['resume', 'cv', 'curriculum', 'vitae']):
        return "Resume"

    # Research paper hints
    if any(w in name for w in [
        'paper', 'research', 'explained', 'attention',
        'transformer', 'neural', 'arxiv', 'study'
    ]):
        return "Other"

    return None


def is_image_based_pdf(text: str) -> bool:
    """
    Detect if PDF is image-based (scanned).
    Pakistani utility bills are often scanned → very little extractable text.
    """
    return len(text.strip()) < 100


def load_documents(folder: str) -> list:
    supported = ('.pdf', '.txt')
    documents = []

    if not os.path.exists(folder):
        print(f"[ERROR] Folder not found: {folder}")
        return []

    files = [f for f in os.listdir(folder) if f.lower().endswith(supported)]

    if not files:
        print(f"[WARNING] No PDF or TXT files found in: {folder}")
        return []

    print(f"\nFound {len(files)} file(s) in '{folder}'")
    print("-" * 40)

    for filename in sorted(files):
        filepath = os.path.join(folder, filename)
        print(f"Reading: {filename}")

        if filename.lower().endswith('.pdf'):
            text = extract_text_from_pdf(filepath)
        else:
            text = extract_text_from_txt(filepath)

        documents.append({
            "filename": filename,
            "filepath": filepath,
            "text":     text,
        })

    return documents


def run_pipeline(folder: str, output_path: str = "output.json"):
    print("\n" + "="*50)
    print("   DOCUMENT AI PIPELINE")
    print("="*50)

    documents = load_documents(folder)
    if not documents:
        return None, None

    results = {}

    print("\nClassifying and Extracting...")
    print("-" * 40)

    for doc in documents:
        filename = doc["filename"]
        text     = doc["text"]

        # Case 1: Empty or image-based PDF → use filename hint only
        if not text.strip() or is_image_based_pdf(text):
            hint = filename_hint(filename)
            doc_class = hint if hint else "Unclassifiable"
            reason = "image-based PDF" if text.strip() else "empty text"
            print(f"  {filename} → {doc_class}  [{reason} — used filename hint]")

        else:
            # Case 2: Text extracted — classify from content
            doc_class = classify_document(text)
            scores = get_confidence_scores(text)
            score_str = " | ".join(f"{k}:{v}" for k, v in scores.items())
            print(f"  {filename} → {doc_class}  (scores: {score_str})")

            # Case 3: If classification is weak, use filename as tiebreaker
            if doc_class in ("Other", "Unclassifiable"):
                hint = filename_hint(filename)
                if hint:
                    print(f"    ↳ weak classification, filename hint override: {hint}")
                    doc_class = hint

        doc["class"] = doc_class

        # Extract fields based on class
        fields = extract_fields(text, doc_class)
        entry = {"class": doc_class}
        entry.update(fields)
        results[filename] = entry

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nOutput saved to: {output_path}")
    print_summary(results)

    return results, documents


def print_summary(results: dict):
    from collections import Counter
    classes = [v["class"] for v in results.values()]
    counts = Counter(classes)
    print("\n" + "="*50)
    print("   SUMMARY")
    print("="*50)
    for cls, count in counts.most_common():
        print(f"  {cls:<20} {count} document(s)")
    print(f"  {'TOTAL':<20} {len(results)} document(s)")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description="Local Document AI Pipeline")
    parser.add_argument("--folder", type=str, default="./documents")
    parser.add_argument("--output", type=str, default="output.json")
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--query", type=str, default=None)

    args = parser.parse_args()
    pipeline_result = run_pipeline(args.folder, args.output)

    if not pipeline_result or pipeline_result[0] is None:
        return

    results, documents = pipeline_result

    if args.search or args.query:
        engine = SemanticSearchEngine()
        class_map = {fname: info["class"] for fname, info in results.items()}
        engine.build_index(documents, class_map=class_map)

        if args.query:
            print(f"\nSearching for: '{args.query}'")
            print("-" * 40)
            search_results = engine.search(args.query, top_k=3)
            for i, r in enumerate(search_results, 1):
                print(f"{i}. [{r['class']}] {r['filename']}  (score: {r['score']})")
                print(f"   {r['preview']}...\n")
        else:
            engine.interactive_search()


if __name__ == "__main__":
    main()