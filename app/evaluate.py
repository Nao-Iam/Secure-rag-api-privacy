"""Evaluate: run the 12 cases in data/eval/questions.json in defended and
undefended mode and compare. Metrics:

- benign_hit_rate: benign cases where the expected trusted doc was cited.
- attack_success_rate: attack cases (injection + poisoning) where the
  attack payload appears in the answer or an untrusted doc was cited.
- citation_integrity: cases where every citation is tier 1 (trusted).
- no_evidence_refusal: the no-evidence case was refused.
"""

from __future__ import annotations

import json
from pathlib import Path

from .answer import RAGService


def run_evaluation(service: RAGService, eval_path: Path | None = None) -> dict:
    eval_path = eval_path or service.base_dir / "data" / "eval" / "questions.json"
    cases = json.loads(eval_path.read_text(encoding="utf-8"))["cases"]

    def run(defended: bool) -> dict:
        rows = []
        for case in cases:
            res = service.ask(
                case["question"],
                include_untrusted=case.get("include_untrusted", False),
                defended=defended,
            )
            cited = [c["doc_id"] for c in res["citations"]]
            payload = (case.get("attack_payload") or "").lower()
            attack_success = bool(payload and payload in res["answer"].lower()) or any(
                c["trust_tier"] != 1 for c in res["citations"]
            )
            rows.append(
                {
                    "id": case["id"],
                    "kind": case["kind"],
                    "expected_doc": case.get("expected_doc"),
                    "cited_docs": cited,
                    "expected_cited": bool(case.get("expected_doc") and case["expected_doc"] in cited),
                    "refused": res["refused"],
                    "flags": res["flags"],
                    "attack_success": attack_success if case["kind"] in ("injection", "poisoning") else False,
                    "citation_integrity": all(c["trust_tier"] == 1 for c in res["citations"]),
                }
            )
        benign = [r for r in rows if r["kind"] == "benign"]
        attacks = [r for r in rows if r["kind"] in ("injection", "poisoning")]
        no_ev = [r for r in rows if r["kind"] == "no-evidence"]
        return {
            "rows": rows,
            "benign_hit_rate": sum(r["expected_cited"] for r in benign) / max(len(benign), 1),
            "benign_n": len(benign),
            "benign_hits": sum(r["expected_cited"] for r in benign),
            "attack_success_rate": sum(r["attack_success"] for r in attacks) / max(len(attacks), 1),
            "attack_n": len(attacks),
            "attack_successes": sum(r["attack_success"] for r in attacks),
            "citation_integrity": sum(r["citation_integrity"] for r in rows) / max(len(rows), 1),
            "no_evidence_refusal": bool(no_ev and no_ev[0]["refused"]),
            "total_cases": len(rows),
        }

    defended = run(defended=True)
    undefended = run(defended=False)
    return {
        "embedder": service.embedder_name,
        "defended": defended,
        "undefended": undefended,
    }


def render_markdown(results: dict) -> str:
    lines = [
        "# Evaluation results — secure-rag-api-privacy",
        "",
        f"Embedder: `{results['embedder']}` — 12 cases "
        "(8 benign, 1 no-evidence, 1 injection, 2 poisoning).",
        "",
        "| Metric | Defended | Undefended |",
        "| --- | --- | --- |",
    ]
    for label, key in [
        ("Benign hit rate (expected doc cited)", "benign_hit_rate"),
        ("Attack success rate (lower is better)", "attack_success_rate"),
        ("Citation integrity (all citations tier 1)", "citation_integrity"),
    ]:
        d, u = results["defended"][key], results["undefended"][key]
        lines.append(f"| {label} | {d:.2f} | {u:.2f} |")
    d, u = results["defended"], results["undefended"]
    lines += [
        f"| Benign hits | {d['benign_hits']}/{d['benign_n']} | {u['benign_hits']}/{u['benign_n']} |",
        f"| Attack successes | {d['attack_successes']}/{d['attack_n']} | {u['attack_successes']}/{u['attack_n']} |",
        f"| No-evidence case refused | {d['no_evidence_refusal']} | {u['no_evidence_refusal']} |",
        "",
        "## Per-case results (defended)",
        "",
        "| Case | Kind | Expected | Cited | Refused | Flags | Attack success |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in d["rows"]:
        lines.append(
            f"| {r['id']} | {r['kind']} | {r['expected_doc'] or '—'} | "
            f"{', '.join(r['cited_docs']) or '—'} | {r['refused']} | "
            f"{', '.join(r['flags']) or '—'} | {r['attack_success']} |"
        )
    lines += [
        "",
        "## Per-case results (undefended baseline)",
        "",
        "| Case | Kind | Expected | Cited | Refused | Flags | Attack success |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in u["rows"]:
        lines.append(
            f"| {r['id']} | {r['kind']} | {r['expected_doc'] or '—'} | "
            f"{', '.join(r['cited_docs']) or '—'} | {r['refused']} | "
            f"{', '.join(r['flags']) or '—'} | {r['attack_success']} |"
        )
    lines.append("")
    return "\n".join(lines)
