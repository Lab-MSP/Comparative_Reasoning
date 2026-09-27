"""Convert preference pairs (and optionally reasoning traces) into ms-swift SFT / DPO data.

  SFT     : --format sft                     -> "<answer> Clip 1 </answer>"
  SFT-CoT : --format sft --traces_dir DIR    -> "<think> ... </think> <answer> Clip 1 </answer>"
  DPO     : --format dpo                     -> rejected: the flipped answer
  DPO-CoT : --format dpo --traces_dir DIR    -> rejected: the wrong reasoning trace + flipped answer

--traces_dir holds <emotion>.jsonl files produced by data/reasoning/build_traces.py (or
downloaded from Hugging Face). In CoT mode the question is taken from the traces, and pairs
without a (chosen / rejected) trace are skipped.
"""

import argparse
import json
import os

from data.prompts import (
    SYSTEM_PROMPT,
    answer_from_preference,
    build_question,
    emotion_from_path,
    flip_answer,
    load_pairs,
    pair_key,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--format", "-f", required=True, choices=["sft", "dpo"])
    parser.add_argument(
        "--input",
        "-i",
        nargs="+",
        default=[
            "pairs/msp_podcast/arousal_preference.csv",
            "pairs/msp_podcast/dominance_preference.csv",
            "pairs/msp_podcast/valence_preference.csv",
        ],
        help="Preference CSVs (pass only arousal_preference.csv for the cross-emotion experiment)",
    )
    parser.add_argument("--traces_dir", "-t", default=None, help="Reasoning traces (<emotion>.jsonl); enables CoT")
    parser.add_argument("--audio_dir", "-a", default="/path/to/msp_podcast/Audios")
    parser.add_argument("--output", "-o", required=True, help="Output .jsonl path")
    return parser.parse_args()


def load_traces(traces_dir, emotion):
    path = os.path.join(traces_dir, f"{emotion}.jsonl")
    with open(path) as f:
        return {record["key"]: record for record in map(json.loads, f)}


def main():
    args = parse_args()

    swift_data = []
    for input_file in args.input:
        emotion = emotion_from_path(input_file)
        traces = load_traces(args.traces_dir, emotion) if args.traces_dir else None

        for row in load_pairs(input_file, split="Train"):
            question = build_question(emotion)
            answer = answer_from_preference(row["preference"])
            chosen = f"<answer> {answer} </answer>"
            rejected = f"<answer> {flip_answer(answer)} </answer>"

            if traces is not None:
                trace = traces.get(pair_key(row["sen1"], row["sen2"]))
                if trace is None:
                    continue
                question = trace["question"]
                chosen = f"<think> {trace['reasoning']} </think> {chosen}"
                if args.format == "dpo":
                    if trace["rejected_reasoning"] is None:
                        continue
                    rejected = f"<think> {trace['rejected_reasoning']} </think> <answer> {trace['rejected_answer']} </answer>"

            entry = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"<audio><audio>{question}"},
                    {"role": "assistant", "content": chosen, "loss": True},
                ],
            }
            if args.format == "dpo":
                entry["rejected_response"] = rejected
            entry["audios"] = [
                os.path.abspath(os.path.join(args.audio_dir, row["sen1"])),
                os.path.abspath(os.path.join(args.audio_dir, row["sen2"])),
            ]
            swift_data.append(entry)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        for entry in swift_data:
            f.write(json.dumps(entry) + "\n")
    print(f"Saved {len(swift_data)} entries to {args.output}")


if __name__ == "__main__":
    main()
