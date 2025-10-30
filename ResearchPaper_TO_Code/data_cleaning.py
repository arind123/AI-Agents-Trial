import re

def clean_text(text: str) -> str:
    """Remove unwanted breaks, headers, and spacing."""
    # Remove multiple line breaks
    text = re.sub(r'\n+', '\n', text)

    # Remove page numbers or isolated digits
    text = re.sub(r'\bPage\s*\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # Merge single newlines (within paragraphs)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # Remove excessive spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def normalize_text(text: str) -> str:
    """Normalize quotes, ligatures, etc."""
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = text.replace("“", "\"").replace("”", "\"").replace("’", "'")
    return text

def remove_references(text: str) -> str:
    """Optionally remove reference or bibliography section."""
    parts = re.split(r'\bReferences\b|\bBibliography\b', text, flags=re.IGNORECASE)
    return parts[0] if parts else text