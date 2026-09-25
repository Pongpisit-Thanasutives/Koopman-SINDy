"""Validate the final editorial/layout pass without rerunning any experiment."""
import hashlib
import json
import re
from validate_v2_reporting import historical_package_path


def presentation_normalized(text):
    """Remove only approved float placement controls and the Table 2 heading edit."""
    for environment in ("figure", "table"):
        statement = "        text=text.replace(r'\\begin{" + environment + "}[H]',r'\\begin{" + environment + "}[!htbp]')\n"
        text = text.replace(statement, "")
    text = re.sub(r"(\\begin\{(?:figure|table)\})(?:\[[!Hhtbp]+\])?", r"\1", text)
    text = text.replace(r"\TCH{$N_{\rm seed}$}", r"\TCH{\#Seeds}")
    text = text.replace(r"\TCH{$N_{\mathrm{seed}}$}", r"\TCH{\#Seeds}")
    text = text.replace(r"\TCH{$N_{\mathrm{seeds}}$}", r"\TCH{\#Seeds}")
    # Only these exact old/new presentation assignments are equivalent here.
    text = text.replace('placement = "p" if group == "pde" else "!t"',
                        'placement = "FLOAT_PLACEMENT"')
    text = text.replace('placement = "!htbp"', 'placement = "FLOAT_PLACEMENT"')
    return text


def validate(root, package, check):
    audit = json.loads((root / "results/final_validation/editorial_preservation_v5.json").read_text())
    for name, expected in audit["preserved_sha256"].items():
        path = historical_package_path(package, name)
        check("V5 unchanged numerical source/evidence: " + name,
              path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected)
    for name, record in audit["presentation_only_changes"].items():
        path = package / name
        check("V5 current presentation source: " + name,
              hashlib.sha256(path.read_bytes()).hexdigest() == record["current_sha256"])
        normalized = presentation_normalized(path.read_text()).encode()
        check("V5 source unchanged apart from approved presentation controls: " + name,
              hashlib.sha256(normalized).hexdigest() == record["baseline_normalized_sha256"])

    directory = package / "latex_source"
    def flattened(path):
        def child(match):
            p = directory / match.group(1)
            return flattened(p if p.suffix else p.with_suffix(".tex"))
        return re.sub(r"\\input\{([^}]+)\}", child, path.read_text())
    expanded = flattened(directory / "koopman_sindy_revised.tex")
    tables = re.findall(r"\\begin\{table\}.*?\\end\{table\}", expanded, re.S)
    figures = re.findall(r"\\begin\{figure\}.*?\\end\{figure\}", expanded, re.S)
    check("V5 all 14 tables preserve every numerical token", len(tables) == 14 and
          [re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", table) for table in tables] == audit["table_numeric_tokens"])
    check("V5 all 9 figures preserve content and captions", len(figures) == 9 and
          [hashlib.sha256(presentation_normalized(figure).encode()).hexdigest() for figure in figures]
          == audit["figure_content_sha256"])

    doc = json.loads((root / "results/final_validation/document_validation.json").read_text())
    check("V5 completed PDF/reference audit", doc["passed"])
    for record in doc["documents"]:
        check("V5 audited PDF fingerprint: " + record["file"],
              hashlib.sha256((package / record["file"]).read_bytes()).hexdigest() == record["sha256"])
    for name, expected in doc["compiled_source_sha256"].items():
        check("V5 compiled source fingerprint: " + name,
              hashlib.sha256((package / name).read_bytes()).hexdigest() == expected)
