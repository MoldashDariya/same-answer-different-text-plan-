#!/usr/bin/env python3
"""Extract "before -> after" edit fragments between KK-orig and KK-nat v2.

Outputs (in linguistic/):
  edit_pairs_v1.csv           one row per edit fragment, with empty tag columns for two annotators
  edit_stats_per_answer.csv   per-answer word-level edit statistics
  edit_stats_summary.md       summary by specialty and quality level

Standard library only. Run from the package root:
  python linguistic/extract_edits.py
"""
import csv, difflib, hashlib, re, statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "linguistic"
TOKEN = re.compile(r"[\w\-’'ʼ]+|[^\w\s]", re.U)
CTX = 3


def rd(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))


def toks(s):
    return TOKEN.findall(s)


def is_word(t):
    return bool(re.match(r"\w", t, re.U))


def main():
    items = {(x["answer_id"], x["answer_version"]): x for x in rd(ROOT / "data/items_v1_2.csv")}
    nat = {x["answer_id"]: x for x in rd(ROOT / "data/kk_nat_v2.csv")}
    q = {x["question_id"]: x for x in rd(ROOT / "protocol/questions_skillbridge_v1_2.csv")}

    pairs, stats = [], []
    for aid in sorted(nat):
        orig = items[(aid, "kk")]
        a_txt, b_txt = orig["text"], nat[aid]["text"]
        assert hashlib.sha256(a_txt.encode()).hexdigest() == nat[aid]["kk_orig_sha256"], aid
        a, b = toks(a_txt), toks(b_txt)
        al, bl = [t.lower() for t in a], [t.lower() for t in b]
        sm = difflib.SequenceMatcher(a=al, b=bl, autojunk=False)
        n_words = sum(map(is_word, a))
        matched_words = 0
        k = 0
        ops = {"replace": 0, "delete": 0, "insert": 0}
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == "equal":
                matched_words += sum(map(is_word, a[i1:i2]))
                continue
            before, after = " ".join(a[i1:i2]), " ".join(b[j1:j2])
            # skip pure punctuation edits but count them
            punct_only = not any(map(is_word, a[i1:i2] + b[j1:j2]))
            ops[op] += 1
            k += 1
            pairs.append({
                "edit_id": f"{aid}-E{k:02d}", "answer_id": aid, "question_id": orig["question_id"],
                "specialty": orig["specialty"], "quality_intent": orig["quality_intent"],
                "style_label": orig["style_label"], "in_main": q[orig["question_id"]]["include_in_main"],
                "op": op, "punct_only": "YES" if punct_only else "NO",
                "before": before, "after": after,
                "context_left": " ".join(a[max(0, i1 - CTX):i1]), "context_right": " ".join(a[i2:i2 + CTX]),
                "possible_move_with": "", "tag_annot1": "", "tag_annot2": "", "tag_final": "", "notes": "",
            })
        # flag likely word-order moves: a deleted span that reappears as an inserted span (difflib splits moves)
        mine = [p for p in pairs if p["answer_id"] == aid]
        for p in mine:
            p.setdefault("possible_move_with", "")
        for d in (p for p in mine if p["op"] == "delete"):
            for i in (p for p in mine if p["op"] == "insert"):
                if d["before"].lower() == i["after"].lower() and not i["possible_move_with"]:
                    d["possible_move_with"], i["possible_move_with"] = i["edit_id"], d["edit_id"]
                    break
        changed = n_words - matched_words + sum(map(is_word, [t for op, i1, i2, j1, j2 in sm.get_opcodes() if op == "insert" for t in b[j1:j2]]))
        stats.append({
            "answer_id": aid, "question_id": orig["question_id"], "specialty": orig["specialty"],
            "quality_intent": orig["quality_intent"], "style_label": orig["style_label"],
            "in_main": q[orig["question_id"]]["include_in_main"],
            "n_words_orig": n_words, "n_words_nat": sum(map(is_word, b)),
            "share_orig_words_changed": round(1 - matched_words / n_words, 4) if n_words else 0,
            "word_edit_rate": round(changed / n_words, 4) if n_words else 0,
            "n_edit_fragments": k, "n_replace": ops["replace"], "n_delete": ops["delete"], "n_insert": ops["insert"],
            "length_change_pct": nat[aid]["length_change_pct"],
        })

    with open(OUT / "edit_pairs_v1.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0])); w.writeheader(); w.writerows(pairs)
    with open(OUT / "edit_stats_per_answer.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0])); w.writeheader(); w.writerows(stats)

    def summ(rows):
        s = [r["share_orig_words_changed"] for r in rows]
        e = [r["n_edit_fragments"] for r in rows]
        return f"| {len(rows)} | {st.mean(s):.3f} | {st.median(s):.3f} | {min(s):.3f}–{max(s):.3f} | {st.mean(e):.1f} |"

    lines = ["# Edit statistics KK-orig → KK-nat v2 (automatic, before tagging)", "",
             f"Fragments: {len(pairs)} (of which punctuation-only: {sum(p['punct_only']=='YES' for p in pairs)}). "
             f"Ops: replace {sum(p['op']=='replace' for p in pairs)}, delete {sum(p['op']=='delete' for p in pairs)}, "
             f"insert {sum(p['op']=='insert' for p in pairs)}. Delete+insert pairs flagged as possible word-order moves: "
             f"{sum(1 for p in pairs if p['possible_move_with'] and p['op']=='delete')}.", "",
             "`share` = share of KK-orig word tokens not kept verbatim (case-insensitive word alignment).", "",
             "| Group | n answers | mean share | median | range | mean fragments |", "|---|---|---|---|---|---|",
             "| all " + summ(stats)]
    for key in ("specialty", "quality_intent", "style_label", "in_main"):
        g = defaultdict(list)
        for r in stats:
            g[r[key]].append(r)
        for v in sorted(g):
            lines.append(f"| {key}={v} " + summ(g[v]))
    (OUT / "edit_stats_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
