#!/usr/bin/env python3
"""SkillBridge KZ — запуск LLM-оценщика (Gemini; другие провайдеры добавляются отдельно).

Режимы:
  smoke  — один тестовый запрос на НЕисследовательском тексте. Проверка доступа и ID модели.
  run    — оценка ответов из пакета. Для версии kk_nat нужен --osf-id (после предрегистрации).

Примеры (из корня пакета):
  python runner/run_judge.py smoke --model <ID_МОДЕЛИ>
  python runner/run_judge.py run --model <ID_МОДЕЛИ> --versions ru kk kk_nat --conditions A A_prime --repeats 3 --osf-id <OSF_ID>

Ключи API: GEMINI_API_KEY (gemini), ANTHROPIC_API_KEY (anthropic), OPENAI_BASE_URL + OPENAI_API_KEY (openai_compat,
например Sherkala через vLLM). Нужен Python 3.11+ (google-genai 3.x).
"""
import argparse, csv, datetime as dt, hashlib, json, os, random, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPT_A = ROOT / "protocol/skillbridge_judge_prompt_a_v1_2.md"
PROMPT_AP = ROOT / "protocol/skillbridge_judge_prompt_a_prime_v1_0.md"
QUESTIONS = ROOT / "protocol/questions_skillbridge_v1_2.csv"
ITEMS = ROOT / "data/items_v1_2.csv"
KK_NAT = ROOT / "data/kk_nat_v2.csv"
OUT = ROOT / "runs"
SEED = 20261107
TEMPERATURE = 0.0
MAX_TOKENS = 8192  # с запасом: «думающие» модели тратят часть токенов на рассуждение
MAX_ATTEMPTS = 3
THINKING_LEVEL = os.environ.get("GEMINI_THINKING_LEVEL")  # None = по умолчанию модели; значение фиксируется в предрегистрации
KEYS = ["raw_score_1to5", "final_score_1to5", "addressed_essential_points", "missed_essential_points",
        "safety_cap_applied", "hypothetical_cap_applied", "uncertainty_flag", "uncertainty_reason", "evidence"]

FENCE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.S)
sha = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()
now = lambda: dt.datetime.now(dt.timezone.utc).isoformat()


def text_blocks(path):
    return re.findall(r"```text\n(.*?)```", path.read_text(encoding="utf-8"), re.S)


def load_prompts():
    a = text_blocks(PROMPT_A)
    system_a, user_tpl = a[0].strip(), a[1].strip()
    reminder = text_blocks(PROMPT_AP)[0].strip()
    return {"A": (system_a, "judge_a_v1.2"),
            "A_prime": (system_a + "\n\n" + reminder, "judge_a_prime_v1.0")}, user_tpl


def render_user(tpl, q, answer):
    vals = {"question_id": q["question_id"], "safety_critical": q["safety_critical"],
            "experience_question": q["experience_question"], "question_en": q["question_en"],
            "reference_points_en": q["reference_points_en"], "essential_points": q["essential_points"],
            "student_answer": answer}
    return re.sub(r"\{\{(\w+)\}\}", lambda m: str(vals[m.group(1)]), tpl)


def validate(obj, q):
    if not isinstance(obj, dict) or set(obj) != set(KEYS):
        return "keys"
    for k in ("raw_score_1to5", "final_score_1to5"):
        if not isinstance(obj[k], int) or isinstance(obj[k], bool) or not 1 <= obj[k] <= 5:
            return k
    for k in ("safety_cap_applied", "hypothetical_cap_applied", "uncertainty_flag"):
        if not isinstance(obj[k], bool):
            return k
    ess = {int(x) for x in q["essential_points"].split(";") if x.strip()}
    ad, mi = obj["addressed_essential_points"], obj["missed_essential_points"]
    if not all(isinstance(x, int) for x in ad + mi) or len(set(ad)) != len(ad) or len(set(mi)) != len(mi):
        return "points_type"
    if set(ad) & set(mi) or set(ad) | set(mi) != ess:
        return "points_partition"
    if obj["safety_cap_applied"] and q["safety_critical"] != "1":
        return "safety_cap_not_allowed"
    if obj["hypothetical_cap_applied"] and q["experience_question"] != "1":
        return "hyp_cap_not_allowed"
    exp = min(obj["raw_score_1to5"], 2 if obj["safety_cap_applied"] else 5, 4 if obj["hypothetical_cap_applied"] else 5)
    if obj["final_score_1to5"] != exp:
        return "final_score_rule"
    if not isinstance(obj["evidence"], str) or not obj["evidence"].strip():
        return "evidence"
    return None


# ---------- провайдер: Gemini ----------
class Gemini:
    name = "google_gemini"

    def __init__(self, model):
        from google import genai
        from google.genai import types
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            sys.exit("Нет GEMINI_API_KEY. См. runner/README_RUN.md, шаг 2.")
        self.client, self.types, self.model = genai.Client(api_key=key), types, model
        import google.genai as g
        self.sdk = getattr(g, "__version__", "unknown")

    def settings(self):
        return {"temperature_sent": TEMPERATURE, "temperature_note": "ignored by Gemini 3.6+ per Google docs (2026-11-07)",
                "max_output_tokens": MAX_TOKENS, "seed": SEED, "thinking_level": THINKING_LEVEL or "model_default",
                "response_mime_type": "application/json", "sdk": f"google-genai {self.sdk}"}

    def call(self, system, user):
        cfg = self.types.GenerateContentConfig(system_instruction=system, temperature=TEMPERATURE,
                                               max_output_tokens=MAX_TOKENS, seed=SEED,
                                               response_mime_type="application/json",
                                               **({"thinking_config": self.types.ThinkingConfig(thinking_level=THINKING_LEVEL)}
                                                  if THINKING_LEVEL else {}))
        r = self.client.models.generate_content(model=self.model, contents=user, config=cfg)
        raw = r.model_dump(mode="json") if hasattr(r, "model_dump") else str(r)
        return (r.text or ""), getattr(r, "response_id", None), getattr(r, "model_version", None), raw


class Anthropic:
    name = "anthropic"

    def __init__(self, model):
        import anthropic
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("Нет ANTHROPIC_API_KEY.")
        self.client, self.model, self.sdk = anthropic.Anthropic(), model, anthropic.__version__

    # SDK 2.x убрал temperature из сигнатуры, а API-версия 2026-10-21 его отклоняет (400, проверено 2026-11-07).
    # API-версия 2023-06-01 принимает temperature=0, поэтому версия фиксируется явно и пишется в settings.
    API_VERSION = "2023-06-01"

    def settings(self):
        return {"temperature": TEMPERATURE, "max_tokens": 1024, "anthropic_version": self.API_VERSION,
                "sdk": f"anthropic {self.sdk}"}

    def call(self, system, user):
        r = self.client.messages.create(model=self.model, system=system, max_tokens=1024,
                                        messages=[{"role": "user", "content": user}],
                                        extra_body={"temperature": TEMPERATURE},
                                        extra_headers={"anthropic-version": self.API_VERSION})
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        return text, r.id, r.model, r.model_dump(mode="json")


class OpenAICompat:  # Sherkala/KAZ-LLM через vLLM или любой OpenAI-совместимый сервер: OPENAI_BASE_URL, OPENAI_API_KEY
    name = "openai_compatible"

    def __init__(self, model):
        import openai
        self.base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.client, self.model, self.sdk = openai.OpenAI(base_url=self.base), model, openai.__version__

    def settings(self):
        return {"temperature": TEMPERATURE, "max_tokens": 1024, "seed": SEED, "base_url": self.base,
                "sdk": f"openai {self.sdk}"}

    def call(self, system, user):
        r = self.client.chat.completions.create(model=self.model, temperature=TEMPERATURE, max_tokens=1024, seed=SEED,
                                                messages=[{"role": "system", "content": system},
                                                          {"role": "user", "content": user}])
        return (r.choices[0].message.content or ""), r.id, r.model, r.model_dump(mode="json")


class Mock:  # для проверки скрипта без API: --provider mock
    name, sdk = "mock", "none"

    def __init__(self, model): self.model = model
    def settings(self): return {"temperature": TEMPERATURE, "mock": True}

    def call(self, system, user):
        ess = re.search(r"essential_points.*?: (.*)", user).group(1).strip().split(";")
        o = {"raw_score_1to5": 3, "final_score_1to5": 3, "addressed_essential_points": [int(ess[0])],
             "missed_essential_points": [int(x) for x in ess[1:]], "safety_cap_applied": False,
             "hypothetical_cap_applied": False, "uncertainty_flag": False, "uncertainty_reason": None,
             "evidence": "mock"}
        return json.dumps(o), "mock-id", self.model, {"mock": True}


def judge(prov, system, user, q):
    attempts = []
    for i in range(1, MAX_ATTEMPTS + 1):
        t0 = time.time()
        try:
            text, rid, mv, raw = prov.call(system, user)
            err = None
            fence = FENCE.fullmatch(text.strip())  # предрегистрированное правило: снимается только одна обёртка ```json ... ```
            try:
                obj = json.loads(fence.group(1) if fence else text)
                err = validate(obj, q)
            except json.JSONDecodeError:
                obj, err = None, "json"
            except (TypeError, AttributeError, ValueError):
                obj, err = None, "schema"
        except Exception as e:  # сетевые/API ошибки тоже логируются
            text, rid, mv, raw, obj, err = "", None, None, {"exception": repr(e)}, None, "api_error"
        attempts.append({"attempt": i, "time_utc": now(), "latency_s": round(time.time() - t0, 2),
                         "request_id": rid, "fence_stripped": bool(text) and FENCE.fullmatch(text.strip()) is not None, "returned_model_version": mv, "error": err,
                         "text": text, "raw_response": raw})
        if err is None:
            return obj, attempts, "ok"
        time.sleep(2 * i)
    # все попытки — сетевые/API ошибки: такой запрос при перезапуске отправляется заново
    return None, attempts, ("api_error" if all(x["error"] == "api_error" for x in attempts) else "parse_failed")


def load_items(versions):
    q = {x["question_id"]: x for x in csv.DictReader(open(QUESTIONS, encoding="utf-8-sig"))}
    rows = []
    for x in csv.DictReader(open(ITEMS, encoding="utf-8-sig")):
        if x["answer_version"] in versions:
            rows.append((x["answer_id"], x["question_id"], x["answer_version"], x["text"]))
    if "kk_nat" in versions:
        for x in csv.DictReader(open(KK_NAT, encoding="utf-8-sig")):
            rows.append((x["answer_id"], x["question_id"], "kk_nat", x["text"]))
    return q, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["smoke", "run"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--provider", default="gemini", choices=["gemini", "anthropic", "openai_compat", "mock"])
    ap.add_argument("--versions", nargs="+", default=["ru", "kk", "kk_nat"])
    ap.add_argument("--conditions", nargs="+", default=["A", "A_prime"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--osf-id", default=None)
    ap.add_argument("--limit", type=int, default=None, help="только первые N запросов (для проверки)")
    a = ap.parse_args()

    prov = {"gemini": Gemini, "anthropic": Anthropic, "openai_compat": OpenAICompat, "mock": Mock}[a.provider](a.model)
    prompts, user_tpl = load_prompts()
    OUT.mkdir(exist_ok=True)

    if a.mode == "smoke":
        q = {"question_id": "SMOKE", "safety_critical": "0", "experience_question": "0",
             "question_en": "What would you do first if a printer in the office stops printing?",
             "reference_points_en": "Check power and cable | Check paper and error lights | Check the print queue | Ask a supervisor if unresolved",
             "essential_points": "1;3"}
        answer = "I would check that it is switched on and see if jobs are stuck in the queue."  # не из исследования
        system, pv = prompts["A"]
        user = render_user(user_tpl, q, answer)
        obj, att, status = judge(prov, system, user, q)
        rec = {"mode": "smoke", "date_utc": now(), "provider": prov.name, "requested_model": a.model,
               "settings": prov.settings(), "status": status, "result": obj, "attempts": att}
        p = OUT / f"smoke_{a.provider}_{a.model.replace('/', '_')}_{dt.date.today()}.json"
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        last = att[-1]
        print(f"status={status}\nrequest_id={last['request_id']}\nreturned_model_version={last['returned_model_version']}\nfile={p}")
        print("Перепишите эти три значения в protocol/MODEL_ACCESS_CHECK.md")
        return

    if "kk_nat" in a.versions and not a.osf_id and a.provider != "mock":
        sys.exit("Версию kk_nat можно запускать только после предрегистрации: добавьте --osf-id <ID>.")

    q, items = load_items(a.versions)
    jobs = [(c, aid, qid, ver, txt, r) for (aid, qid, ver, txt) in items for c in a.conditions for r in range(1, a.repeats + 1)]
    random.Random(SEED).shuffle(jobs)  # порядок фиксирован сидом; версии одного ответа — в разных запросах
    if a.limit:
        jobs = jobs[:a.limit]
    out = OUT / f"run_{a.provider}_{a.model.replace('/', '_')}.jsonl"
    done = set()
    if out.exists():
        for line in open(out, encoding="utf-8"):
            r = json.loads(line)
            if r["status"] == "api_error":
                continue  # повторяется; в анализе берётся последняя запись по ключу
            done.add((r["rubric_condition"], r["answer_id"], r["answer_version"], r["repeat"]))
    print(f"заданий: {len(jobs)}, уже сделано: {len(done)} → {out}")
    with open(out, "a", encoding="utf-8") as f:
        for n, (c, aid, qid, ver, txt, rep) in enumerate(jobs, 1):
            if (c, aid, ver, rep) in done:
                continue
            system, pv = prompts[c]
            user = render_user(user_tpl, q[qid], txt)
            obj, att, status = judge(prov, system, user, q[qid])
            rec = {"answer_id": aid, "question_id": qid, "answer_version": ver, "text_source": "synthetic_text",
                   "rubric_condition": c, "prompt_version": pv, "prompt_sha256": sha(system + "\n\n" + user),
                   "input_text_sha256": sha(txt), "repeat": rep, "run_date": now(), "provider": prov.name,
                   "judge_model": a.model, "returned_model_version": att[-1]["returned_model_version"],
                   "settings": prov.settings(), "osf_id": a.osf_id, "status": status,
                   "raw_score_1to5": obj["raw_score_1to5"] if obj else None,
                   "final_score_1to5": obj["final_score_1to5"] if obj else None,
                   "parsed": obj, "attempts": att}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            if n % 20 == 0:
                print(f"{n}/{len(jobs)}")
    print("готово:", out)


if __name__ == "__main__":
    main()
