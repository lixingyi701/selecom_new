"""
Convert NQ validation parquet files to SeleCom JSONL format.

Input format (HuggingFace NQ):
  question: {text, tokens}
  document: {html, title, tokens:{token, is_html, start_byte, end_byte}, url}
  annotations: {
    short_answers: [{text, start_byte, end_byte, ...}],  # extractive spans
    yes_no_answer: int array (-1 = no, 0 = NO, 1 = YES),
    long_answer: [{candidate_index, ...}],                # paragraph-level
  }

Output format (SeleCom):
  {"question": str, "answer": [str, ...], "documents": ["Title: ...\nContent: ..."]}

Inclusion rules (matching standard NQ ODQA practice):
  - Keep: questions with extractive short answers (e.g. "Neil Armstrong")
  - Keep: questions with yes/no answers only → answer = "yes" / "no"
  - Skip: questions with only a long paragraph answer (EM/F1 not meaningful)
  - Skip: questions where all 5 annotators gave no answer
"""

import glob
import json

import numpy as np
import pandas as pd
from tqdm import tqdm


WINDOW_BYTES = 4000   # bytes on each side of the answer span to include
MAX_TEXT_TOKENS = 400  # hard cap on text tokens in the passage


def extract_passage(document: dict, ans_start: int, ans_end: int) -> str:
    """Extract a clean text passage around the answer span from NQ document tokens."""
    tok = document["tokens"]
    texts = tok["token"]
    is_html = tok["is_html"]
    starts = tok["start_byte"]
    ends = tok["end_byte"]

    if ans_start >= 0:
        win_lo = max(0, ans_start - WINDOW_BYTES)
        win_hi = ans_end + WINDOW_BYTES
        mask = (~is_html) & (starts >= win_lo) & (ends <= win_hi)
    else:
        mask = ~is_html

    selected = texts[mask]
    if len(selected) > MAX_TEXT_TOKENS:
        selected = selected[:MAX_TEXT_TOKENS]

    return " ".join(selected)


YESNO_MAP = {0: "no", 1: "yes"}  # NQ encoding: -1=N/A, 0=NO, 1=YES


def extract_answers(annotations: dict):
    """Return (answers, first_answer_start_byte, first_answer_end_byte).

    Priority: extractive short answers > yes/no answers.
    Returns ([], -1, -1) if neither exists (long-only or no answer).
    """
    sa = annotations["short_answers"]
    yn = annotations["yes_no_answer"]
    answers = []
    ans_start, ans_end = -1, -1

    # Collect extractive short answers
    for item in sa:
        texts = item["text"]
        s_bytes = item["start_byte"]
        e_bytes = item["end_byte"]

        if len(texts) > 0:
            for t in texts:
                t = str(t).strip()
                if t:
                    answers.append(t)
            if ans_start < 0 and len(s_bytes) > 0:
                ans_start = int(s_bytes[0])
                ans_end = int(e_bytes[0])

    # Fall back to yes/no if no extractive answers
    if not answers:
        for v in yn:
            v = int(v)
            if v in YESNO_MAP:
                answers.append(YESNO_MAP[v])

    # Deduplicate while preserving order
    seen = set()
    unique_answers = []
    for a in answers:
        if a not in seen:
            seen.add(a)
            unique_answers.append(a)

    return unique_answers, ans_start, ans_end


def convert(data_dir: str, output_path: str):
    files = sorted(glob.glob(f"{data_dir}/validation-*.parquet"))
    print(f"Found {len(files)} parquet files")

    results = []
    skipped_long_only = 0
    skipped_no_answer = 0

    for fpath in tqdm(files, desc="Parquet files"):
        df = pd.read_parquet(fpath)

        for i in range(len(df)):
            row = df.iloc[i]
            ann = row["annotations"]

            question = row["question"]["text"]
            answers, ans_start, ans_end = extract_answers(ann)

            if not answers:
                # Distinguish: has a long answer vs. truly unanswerable
                has_long = any(item["candidate_index"] != -1 for item in ann["long_answer"])
                if has_long:
                    skipped_long_only += 1
                else:
                    skipped_no_answer += 1
                continue

            doc = row["document"]
            title = doc.get("title", "")
            passage = extract_passage(doc, ans_start, ans_end)
            document_str = f"Title: {title}\nContent: {passage}"

            results.append({
                "question": question,
                "answer": answers,
                "documents": [document_str],
            })

    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nDone.")
    print(f"  Kept (short + yes/no answers): {len(results)}")
    print(f"  Skipped (long paragraph only): {skipped_long_only}")
    print(f"  Skipped (no answer at all):    {skipped_no_answer}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    convert(
        data_dir=os.path.join(script_dir, "eval"),
        output_path=os.path.join(script_dir, "eval", "nq_eval.jsonl"),
    )
