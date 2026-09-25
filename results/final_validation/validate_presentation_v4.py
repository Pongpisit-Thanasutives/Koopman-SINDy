"""Independent saved-file and table-formatting checks for the final presentation pass."""
import ast
import hashlib
import json
import re
from validate_v2_reporting import historical_package_path


def validate(root, package, check):
    audit = json.loads((root / "results/final_validation/presentation_preservation.json").read_text())
    for filename, expected in audit["preserved_sha256"].items():
        path = historical_package_path(package, filename)
        check("V4 preserved evidence/source: " + filename, path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected)
    for filename, expected in audit["formatted_output_sha256"].items():
        check("V4 current formatted output: " + filename, hashlib.sha256((package / filename).read_bytes()).hexdigest() == expected)
    for filename, record in audit["reporting_only_changes"].items():
        path = root / filename
        check("V4 reporting source: " + filename, hashlib.sha256(path.read_bytes()).hexdigest() == record["current_source_sha256"])
        tree = ast.parse(path.read_text())
        tree.body = [node for node in tree.body if not isinstance(node, ast.FunctionDef) or node.name not in record["changed_reporting_functions"]]
        check("V4 unchanged module outside reporting functions: " + filename,
              hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest() == record["unchanged_module_ast_sha256"])
        if "numerical_prefix" in record:
            cfg = record["numerical_prefix"]
            tree = ast.parse(path.read_text())
            node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == cfg["function"])
            prefix = ast.Module(body=node.body[:cfg["statement_count"]], type_ignores=[])
            check("V4 unchanged numerical prefix: " + filename,
                  hashlib.sha256(ast.dump(prefix, include_attributes=False).encode()).hexdigest() == cfg["sha256"])

    directory = package / "latex_source"
    def flattened(path):
        def child(match):
            p = directory / match.group(1)
            if not p.suffix:
                p = p.with_suffix(".tex")
            return flattened(p)
        return re.sub(r"\\input\{([^}]+)\}", child, path.read_text())
    expanded = flattened(directory / "koopman_sindy_revised.tex")
    tables = re.findall(r"\\begin\{table\}.*?\\end\{table\}", expanded, re.S)
    figures = re.findall(r"\\begin\{figure\}.*?\\end\{figure\}", expanded, re.S)
    check("V4 all 14 table numeric-token sequences preserved", len(tables) == 14
          and [re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", table) for table in tables] == audit["table_numeric_tokens"])
    check("All 9 figure environments match current audited sources", len(figures) == 9
          and [hashlib.sha256(figure.encode()).hexdigest() for figure in figures] == audit["figure_environment_sha256"])

    specs = [
        ("revision_pod_attribution.tex", [3, 3, 3], [(3, True), (4, False), (5, True)]),
        ("revision_paired_wins.tex", [2, 2, 2, 2, 2], [(3, True), (4, True)]),
        ("revision_nonoracle_tv_table.tex", [3, 3, 4, 4], [(3, True), (4, True), (5, False)]),
        ("revision_weak_comparison.tex", [4, 6], [(1, True), (2, False), (3, True), (4, True)]),
    ]
    for filename, sizes, columns in specs:
        text = (directory / "tables" / filename).read_text()
        lines = [line for line in text.splitlines() if "&" in line and re.search(r"\d+\.\d+", line)
                 and not any(token in line for token in ("caption", "TCH", "multicolumn"))]
        check("V4 formatted table row count: " + filename, len(lines) == sum(sizes))
        offset = 0
        for block, size in enumerate(sizes):
            rows = [line.split("&") for line in lines[offset:offset + size]]
            for column, maximize in columns:
                # The first numeric token is the displayed mean, error, win
                # percentage, or exact-support numerator. SEs are not ranked.
                values = [float(re.search(r"\d+(?:\.\d+)?", row[column]).group()) for row in rows]
                best = max(values) if maximize else min(values)
                for index, (row, value) in enumerate(zip(rows, values)):
                    bold = bool(re.search(r"\\(?:mathbf|textbf)\{", row[column]))
                    check(f"V4 correct displayed-value emphasis {filename}/{block}/{index}/{column}", bold == (value == best))
            offset += size
    for filename in ("revision_ode_system.tex", "revision_pde_system.tex", "revision_tv_high_noise.tex", "revision_strategy_ablation.tex"):
        check("V4 requested caption wording: " + filename,
              "not statistical significance" not in (directory / "tables" / filename).read_text())
