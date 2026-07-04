import argparse
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIB_PATH = ROOT / "paper" / "mypaper" / "references.bib"
DEFAULT_REPORT_PATH = ROOT / "paper" / "mypaper" / "reference_verification_report.md"
DEFAULT_CSV_PATH = ROOT / "results" / "reference_verification.csv"
NO_DOI_VENUE_PATTERNS = [
    "international conference on learning representations",
    "advances in neural information processing systems",
    "international conference on machine learning",
    "proceedings of machine learning research",
    "relational representation learning workshop",
    "openreview",
]

FIELD_PATTERN = re.compile(r"(\w+)\s*=\s*[\{\"](.*?)[\}\"]\s*,?\s*$", re.IGNORECASE)
ENTRY_PATTERN = re.compile(r"@(\w+)\s*\{\s*([^,]+),")


def split_entries(text):
    entries = []
    current = []
    depth = 0
    inside = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("@") and not inside:
            current = [line]
            depth = line.count("{") - line.count("}")
            inside = True
            continue
        if inside:
            current.append(line)
            depth += line.count("{") - line.count("}")
            if depth <= 0:
                entries.append("\n".join(current))
                current = []
                inside = False
    return entries


def parse_entry(entry_text):
    header = ENTRY_PATTERN.search(entry_text)
    if header is None:
        return None
    entry_type = header.group(1).lower()
    key = header.group(2).strip()
    fields = {}
    for line in entry_text.splitlines()[1:]:
        stripped = line.strip().rstrip(",")
        if not stripped or stripped == "}":
            continue
        match = FIELD_PATTERN.match(stripped)
        if match:
            fields[match.group(1).lower()] = match.group(2).strip()
    return {"key": key, "entry_type": entry_type, "fields": fields}


def venue_field(entry_type, fields):
    if "journal" in fields and fields["journal"]:
        return "journal", fields["journal"]
    if "booktitle" in fields and fields["booktitle"]:
        return "booktitle", fields["booktitle"]
    if entry_type == "misc" and "howpublished" in fields and fields["howpublished"]:
        return "howpublished", fields["howpublished"]
    return "", ""


def venue_without_doi_convention(venue_value):
    venue_lower = venue_value.lower()
    return any(pattern in venue_lower for pattern in NO_DOI_VENUE_PATTERNS)


def classify_entry(entry):
    fields = entry["fields"]
    venue_name, venue_value = venue_field(entry["entry_type"], fields)
    has_title = bool(fields.get("title"))
    has_author = bool(fields.get("author"))
    has_year = bool(fields.get("year"))
    has_venue = bool(venue_value)
    has_doi = bool(fields.get("doi"))
    is_preprint = "arxiv" in venue_value.lower() or "arxiv" in fields.get("journal", "").lower()
    no_doi_expected = has_venue and venue_without_doi_convention(venue_value)
    if has_title and has_author and has_year and has_venue and (has_doi or is_preprint or no_doi_expected):
        status = "verified"
    elif has_title and has_author and has_year and has_venue:
        status = "missing_doi"
    elif has_title and has_author and has_year:
        status = "missing_venue"
    else:
        status = "incomplete"
    return {
        "key": entry["key"],
        "entry_type": entry["entry_type"],
        "title": fields.get("title", ""),
        "year": fields.get("year", ""),
        "venue_field": venue_name,
        "venue_value": venue_value,
        "doi": fields.get("doi", ""),
        "has_title": has_title,
        "has_author": has_author,
        "has_year": has_year,
        "has_venue": has_venue,
        "has_doi": has_doi,
        "is_preprint": is_preprint,
        "no_doi_expected": no_doi_expected,
        "status": status,
    }


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "key",
        "entry_type",
        "title",
        "year",
        "venue_field",
        "venue_value",
        "doi",
        "has_title",
        "has_author",
        "has_year",
        "has_venue",
        "has_doi",
        "is_preprint",
        "no_doi_expected",
        "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(rows, path, source_bib):
    path.parent.mkdir(parents=True, exist_ok=True)
    total = len(rows)
    verified = sum(row["status"] == "verified" for row in rows)
    missing_doi = [row for row in rows if row["status"] == "missing_doi" and not row["is_preprint"] and not row["no_doi_expected"]]
    no_doi_expected = [row for row in rows if row["no_doi_expected"] and not row["has_doi"]]
    missing_venue = [row for row in rows if row["status"] == "missing_venue"]
    incomplete = [row for row in rows if row["status"] == "incomplete"]
    preprints = [row for row in rows if row["is_preprint"]]
    lines = [
        "# Reference Verification Report",
        "",
        f"- Source bibliography: `{source_bib}`",
        f"- Total entries: {total}",
        f"- Verified entries (venue present and DOI, preprint identifier, or no-DOI venue convention recognized): {verified}",
        f"- Non-preprint entries missing DOI where DOI is expected: {len(missing_doi)}",
        f"- Entries in venues that typically do not assign DOI: {len(no_doi_expected)}",
        f"- Entries missing venue: {len(missing_venue)}",
        f"- Incomplete entries: {len(incomplete)}",
        f"- Preprints: {len(preprints)}",
        "",
        "## Entries missing DOI where DOI is expected",
        "",
    ]
    if missing_doi:
        for row in missing_doi:
            lines.append(f"- `{row['key']}` | {row['entry_type']} | {row['venue_value']} | {row['year']}")
    else:
        lines.append("- None")
    lines.extend(["", "## Entries in venues without a DOI convention", ""])
    if no_doi_expected:
        for row in no_doi_expected:
            lines.append(f"- `{row['key']}` | {row['entry_type']} | {row['venue_value']} | {row['year']}")
    else:
        lines.append("- None")
    lines.extend(["", "## Entries missing venue", ""])
    if missing_venue:
        for row in missing_venue:
            lines.append(f"- `{row['key']}` | {row['entry_type']} | {row['year']}")
    else:
        lines.append("- None")
    lines.extend(["", "## Incomplete entries", ""])
    if incomplete:
        for row in incomplete:
            lines.append(f"- `{row['key']}` | title={row['has_title']} author={row['has_author']} year={row['has_year']} venue={row['has_venue']}")
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bib", default=str(DEFAULT_BIB_PATH))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--csv", default=str(DEFAULT_CSV_PATH))
    return parser.parse_args()


def main():
    args = parse_args()
    bib_path = Path(args.bib)
    text = bib_path.read_text(encoding="utf-8")
    parsed = [parse_entry(entry) for entry in split_entries(text)]
    rows = [classify_entry(entry) for entry in parsed if entry is not None]
    csv_path = Path(args.csv)
    report_path = Path(args.report)
    write_csv(rows, csv_path)
    write_report(rows, report_path, bib_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
