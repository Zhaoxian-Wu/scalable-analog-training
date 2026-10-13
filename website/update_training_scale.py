"""Export the manuscript's training-scale table for the interactive homepage.

Run from any directory with: python website/update_training_scale.py
The generated JavaScript also works when index.html is opened directly.
"""
import ast
import argparse
from collections import Counter
import json
from pathlib import Path
import re

WEBSITE = Path(__file__).resolve().parent


def research_root():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root', type=Path,
                        help='Research workspace containing tex/iclr2026 and scripts/')
    args = parser.parse_args()
    candidates = [args.research_root.resolve()] if args.research_root else WEBSITE.parents
    for candidate in candidates:
        if (candidate / 'tex/iclr2026/main.tex').is_file():
            return candidate
    parser.error('Manuscript not found. Pass --research-root /path/to/analog-transformer; '
                 'the standalone GitHub checkout can use the bundled chart data without exporting.')


def bib_field(entry, name):
    match = re.search(r"\b" + name + r"\s*=\s*\{", entry, re.I)
    if not match:
        return ""
    start = match.end()
    depth = 1
    for end in range(start, len(entry)):
        if entry[end] == "{":
            depth += 1
        elif entry[end] == "}":
            depth -= 1
        if depth == 0:
            return entry[start:end]
    raise ValueError(f"Unclosed BibTeX field: {name}")


def plain(value):
    accents = {
        r'{\"a}': 'ä', r'{\"o}': 'ö', r'{\"u}': 'ü',
        r'{\"A}': 'Ä', r'{\"O}': 'Ö', r'{\"U}': 'Ü',
    }
    for latex, unicode_value in accents.items():
        value = value.replace(latex, unicode_value)
    return re.sub(r"[{}]", "", value).replace(r"\&", "&").strip()


def number(value):
    match = re.fullmatch(r"\$([\d.]+)\$([kMB]?)", value.strip())
    if not match:
        raise ValueError(f"Unsupported table number: {value}")
    return round(float(match[1]) * {"": 1, "k": 1e3, "M": 1e6, "B": 1e9}[match[2]])


def export(root):
    bib = {}
    for path in (root / "tex/iclr2026/bibs").glob("*.bib"):
        for entry in re.split(r"(?m)(?=^@)", path.read_text()):
            match = re.match(r"@\w+\{([^,]+),", entry)
            if match:
                bib[match[1]] = entry

    source = (root / "tex/iclr2026/main.tex").read_text()
    table = source.split(r"\begin{tabular}{lcccrrr}", 1)[1].split(r"\end{tabular}", 1)[0]
    studies = []
    methods = {
        "rasch2024fast": "AGAD", "wu2025analog": "Residual Learning",
        "gokmen2020tiki": "Tiki-Taka", "nandakumar2020mixed": "Analog MP",
        "ankit2020panther": "PANTHER",
    }
    for line in table.splitlines():
        if line.lstrip().startswith("%") or "&" not in line:
            continue
        fields = [field.strip() for field in line.removesuffix(r"\\").split("&")]
        cited = re.search(r"\\cite\{([^}]+)\}", fields[0])
        if not cited:
            continue
        key = cited[1]
        entry = bib[key]
        authors = bib_field(entry, "author").split(" and ")
        names = [plain(author).split(",")[0] if "," in author else plain(author).split()[-1] for author in authors]
        label = " & ".join(names) if len(names) <= 2 else names[0] + " et al."
        label += ", " + plain(bib_field(entry, "year"))
        weights = number(fields[4])
        tokens = number(fields[6])
        # Preserve the plotting script's unrounded counts for these entries.
        if key == "martemucci2025ferroelectric": weights = 187_800
        if key == "chen2023open" and fields[1] == "H": tokens = 81_200
        studies.append({"id": f"{key}-{fields[1]}", "key": key, "label": label,
                        "title": plain(bib_field(entry, "title")), "method": methods.get(key, ""),
                        "evidence": fields[1], "regime": fields[2], "model": fields[3],
                        "weights": weights, "tokens": tokens, "dataset": fields[5],
                        "partialMappingExcluded": "*" in fields[0]})

    studies.append({"id": "this-work", "key": "this-work", "label": "This work · Wu et al., 2026",
                    "title": "Making Analog Training Scale: Co-Designing Mapping, Optimizer, and Converters",
                    "method": "This work", "evidence": "S", "regime": "F", "model": "Transformer",
                    "weights": 123_532_032, "tokens": 2_467_430_400, "dataset": "OpenWebText",
                    "partialMappingExcluded": False})

    # Compare multisets rather than relying on row order: coincident points have
    # distinct citations, and the plotting script is deliberately unlabeled.
    tree = ast.parse((root / "scripts/plot_prior_training_scale.py").read_text())
    plotted = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("PRIOR_STUDIES", "THIS_WORK"):
                plotted[name] = ast.literal_eval(node.value)
    assert Counter((s["weights"], s["tokens"]) for s in studies[:-1]) == Counter((w, t) for _, w, t in plotted["PRIOR_STUDIES"]), "Table and figure coordinates differ; review the source data before exporting."
    assert (studies[-1]["weights"], studies[-1]["tokens"]) == plotted["THIS_WORK"]
    serialized = json.dumps(studies, ensure_ascii=False, indent=2)
    output = WEBSITE / "assets/training-scale-data.js"
    output.write_text("// Generated from the manuscript table; refresh with website/update_training_scale.py.\nwindow.trainingScaleStudies = " + serialized + ";\n")
    (WEBSITE.parent / "slides/components/trainingScaleData.json").write_text(serialized + "\n")
    print(f"Exported {len(studies)} studies; all coordinates match the manuscript figure.")


if __name__ == "__main__":
    export(research_root())
