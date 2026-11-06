# Same Answer, Different Text? — analysis plan

Preregistered analysis plan, prompts and code for the study **"Same Answer, Different Text? Naturalness Confounds in Synthetic Kazakh–Russian Benchmarks for LLM Judges"**.

The study asks a single question. Does an LLM judge's rubric score for a Kazakh answer change when a native speaker makes the text more natural without changing its content?

**Author:** Dariya Moldash (Kazakh-British Technical University)

## Contents

| Path | What it is |
|---|---|
| `PREREGISTRATION.md` | The full analysis plan: hypotheses, design, judges and settings, analysis, exclusions, limitations |
| `HASHES.txt` | SHA-256 hashes of the frozen data files and of the code at registration time |
| `protocol/skillbridge_judge_prompt_a_v1_2.md` | Judge prompt, condition A |
| `protocol/skillbridge_judge_prompt_a_prime_v1_0.md` | The one sentence added in condition A′ |
| `runner/run_judge.py` | Sends every answer version to each judge as a separate request and logs raw responses |
| `analysis/analyze.py` | Builds all pre-specified tables and the main figure from the raw run files with one command |
| `linguistic/extract_edits.py` | Extracts the post-editing edits (KK-orig → KK-nat) for the exploratory analysis |

## What is not in this repository

The study data are not published here until the ethics decision. This covers the synthetic answers, the native-speaker post-edits, the meaning checks, the human ratings, and the questions and rubric. `HASHES.txt` fixes their exact versions.

## Notes

- The code was frozen together with the plan, so some internal references in it point to the working package rather than to this repository. In particular:
  - `protocol/OSF_PREREGISTRATION_DRAFT.md` refers to `PREREGISTRATION.md`;
  - the runner's `--osf-id` argument takes the identifier of this registration (the Zenodo DOI).
- Requirements: Python 3.11+, see `runner/requirements.txt`.

## License

Text and code: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Registration

- DOI (v1.0, frozen 2026-11-07): [10.5281/zenodo.24023433](https://doi.org/10.5281/zenodo.24023433)
- `PREREGISTRATION.md` is kept exactly as registered in v1.0.
