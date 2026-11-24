"""Run the locally indexed Wuyue deck and write fresh, isolated reports."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET


DEFAULT_INDEX = Path(__file__).resolve().parents[1] / "references" / "pdk-local.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def replace_once(pattern: str, replacement: str, value: str, label: str) -> str:
    result, count = re.subn(pattern, lambda _: replacement, value, flags=re.M)
    if count != 1:
        raise ValueError(f"Unrecognized {label}: expected one occurrence, found {count}")
    return result


def make_runner(source: Path, kind: str) -> str:
    if kind == "klayout_xml_macro":
        dsl = ET.parse(source).getroot().findtext("text")
        if not dsl:
            raise ValueError("The macro has no DRC text")
        dsl = replace_once(
            r'^\s*report\("drc_result\.lyrdb"\)\s*$',
            'RDBCH = report($report)', dsl, "macro report declaration",
        )
        return "source($input)\n" + dsl + '\nRDBCH.rdb.save($report)\n'
    if kind == "project_opa_runner":
        dsl = source.read_text(encoding="utf-8-sig")
        dsl = replace_once(r'^source\("[^"\r\n]*"\)\s*$', 'source($input)', dsl, "input")
        dsl = replace_once(r'^_ch = report\("[^"\r\n]*"\)\s*$', '_ch = report($report)', dsl, "report")
        return replace_once(
            r'^_ch\.rdb\.save\("[^"\r\n]*"\)\s*$',
            '_ch.rdb.save($report)', dsl, "report save",
        )
    raise ValueError(f"Unsupported profile kind: {kind}")


def read_report(report: Path, expected: set[str]) -> dict:
    root = ET.parse(report).getroot()
    category_root, item_root = root.find("categories"), root.find("items")
    if category_root is None or item_root is None:
        raise ValueError("The report is missing categories or items")
    categories = {
        node.findtext("name").strip(): node.findtext("description") or ""
        for node in category_root.iter("category") if node.findtext("name")
    }
    if set(categories) != expected:
        missing, extra = sorted(expected - set(categories)), sorted(set(categories) - expected)
        raise ValueError(f"Unexpected report categories: missing={missing}, extra={extra}")
    items = [
        {
            "category": (node.findtext("category") or "").strip("'\" "),
            "values": [value.text for value in node.findall("values/value")],
        }
        for node in item_root.iter("item")
    ]
    if any(item["category"] not in categories for item in items):
        raise ValueError("A violation refers to an unknown category")
    counts = Counter(item["category"] for item in items)
    return {
        "rule_categories": len(categories),
        "category_descriptions": categories,
        "violation_count": len(items),
        "violation_counts": dict(sorted(counts.items())),
        "items": items,
    }


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    index_path = args.index.resolve()
    index = json.loads(index_path.read_text(encoding="utf-8-sig"))
    profile = index["drc_profiles"][args.profile]
    source = Path(profile["path"]).resolve()
    executable = args.klayout or Path(index["runtime"]["klayout_executable"])
    gds = args.gds.resolve()
    for path in (gds, source, executable):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not gds.stat().st_size:
        raise ValueError("The input GDS is empty")
    input_sha, rule_sha = sha256(gds), sha256(source)
    if rule_sha != profile["sha256"]:
        raise ValueError("The source deck differs from the PDK index; inspect and update the index before using it")
    runner_text = make_runner(source, profile["kind"])
    expected = set(profile["expected_rule_ids"])
    if not expected or len(expected) != profile["expected_rule_categories"]:
        raise ValueError("The PDK index has an invalid rule list")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = (args.output_dir or gds.parent / "drc_checks" / f"{gds.stem}_{stamp}").resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Use a new or empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    runner, report, log = output / "runner.drc", output / "result.lyrdb", output / "run.log"
    runner.write_text(runner_text, encoding="utf-8")
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    command = [str(executable), "-b", "-z", "-r", str(runner),
               "-rd", f"input={gds}", "-rd", f"report={report}"]
    try:
        process = subprocess.run(
            command, cwd=output, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=args.timeout, creationflags=creation_flags,
        )
        log.write_text(process.stdout + "\nSTDERR:\n" + process.stderr, encoding="utf-8")
        if process.returncode != 0 or not report.is_file() or not report.stat().st_size:
            raise RuntimeError(f"DRC did not finish successfully; inspect {log}")
        if sha256(gds) != input_sha or sha256(source) != rule_sha:
            raise RuntimeError("The input GDS or rule deck changed during the run")
        summary = read_report(report, expected)
        summary.update({
            "status": "passed" if summary["violation_count"] == 0 else "violations",
            "gds": str(gds), "gds_sha256": input_sha,
            "profile": args.profile, "source_runset": str(source), "source_runset_sha256": rule_sha,
            "runner": str(runner), "runner_sha256": sha256(runner),
            "index": str(index_path), "index_sha256": sha256(index_path),
            "report": str(report), "log": str(log), "runtime": str(executable),
            "checked_at_utc": stamp,
            "scope_note": profile["scope_note"],
        })
        write_json(output / "result.json", summary)
        print(f"{args.profile}: {summary['rule_categories']} categories, {summary['violation_count']} violations")
        print(f"Report: {output / 'result.json'}")
        return 0 if not summary["violation_count"] else 1
    except Exception as error:
        write_json(output / "execution_error.json", {
            "status": "execution_failed", "error": str(error),
            "gds": str(gds), "gds_sha256": input_sha, "profile": args.profile,
        })
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gds", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--profile", choices=("common", "opa-port"), default="common")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--klayout", type=Path)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    try:
        return run(args)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
