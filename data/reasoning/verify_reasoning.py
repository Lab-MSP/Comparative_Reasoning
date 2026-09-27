"""Parse generated traces and keep only those whose final answer is consistent with the mode.

  blind / guided : keep traces whose answer matches the ground truth
  wrong          : keep traces whose answer is the flipped (wrong) answer

Output: <output> JSON mapping "<sen1>_<sen2>" -> {"reasoning", "answer", "audios"}.
"""

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

from data.prompts import EMOTIONS, answer_from_preference, flip_answer, load_pairs, pair_key

# A header the reasoning LLM sometimes prepends to its answer; removed from the traces.
UNINTENDED_HEADER = "Comparative reasoning grounded in the provided audio clips"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--emotion_type", "-e", required=True, choices=EMOTIONS)
    parser.add_argument("--mode", "-m", default="blind", choices=["blind", "guided", "wrong"])
    parser.add_argument("--pairs_dir", default="pairs/msp_podcast")
    parser.add_argument("--trace_dir", default=None, help="Default: outputs/reasoning/<mode>/<emotion_type>")
    parser.add_argument("--output", "-o", default=None, help="Default: outputs/reasoning/verified/<mode>/<emotion_type>.json")
    args = parser.parse_args()
    if args.trace_dir is None:
        args.trace_dir = f"outputs/reasoning/{args.mode}/{args.emotion_type}"
    if args.output is None:
        args.output = f"outputs/reasoning/verified/{args.mode}/{args.emotion_type}.json"
    return args


def parse_trace(content: str) -> tuple[str, str]:
    """Split a generated trace into (reasoning, answer)."""
    match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        reasoning = content[: match.start()].strip()
    else:
        answer, reasoning = "", content

    reasoning = reasoning.replace("<think>", "").replace("</think>", "").strip()
    reasoning = reasoning.replace(f"{UNINTENDED_HEADER}:", "").strip()
    reasoning = reasoning.replace(UNINTENDED_HEADER, "").strip()
    return reasoning, answer


def main():
    args = parse_args()
    rows = load_pairs(os.path.join(args.pairs_dir, f"{args.emotion_type}_preference.csv"), split="Train")

    stats = Counter()
    verified = {}
    for row in rows:
        key = pair_key(row["sen1"], row["sen2"])
        trace_file = Path(args.trace_dir) / f"{key}.txt"
        if not trace_file.exists():
            stats["missing"] += 1
            continue

        reasoning, answer = parse_trace(trace_file.read_text())
        if answer not in ("Clip 1", "Clip 2"):
            stats["invalid answer"] += 1
            continue

        expected = answer_from_preference(row["preference"])
        if args.mode == "wrong":
            expected = flip_answer(expected)
        if answer != expected:
            stats["inconsistent answer"] += 1
            continue

        verified[key] = {"reasoning": reasoning, "answer": answer, "audios": [row["sen1"], row["sen2"]]}

    print(f"{args.mode}/{args.emotion_type}: {len(verified)}/{len(rows)} verified ({dict(stats)})")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(verified, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
