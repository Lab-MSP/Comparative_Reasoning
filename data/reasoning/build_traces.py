"""Merge the verified blind / guided / wrong traces into one trace file per attribute.

This is the format of the released reasoning traces (Hugging Face) and the input of
data/post_process/make_swift_data.py. One JSON object per line, in the order of the
Train pairs in <emotion>_preference.csv:

  {"key", "emotion", "sen1", "sen2", "preference",
   "question",                                     # question asked to the audio LM (Clip 1 = sen1, Clip 2 = sen2)
   "answer", "reasoning", "reasoning_source",      # chosen trace; source is "blind" or "guided"
   "rejected_answer", "rejected_reasoning"}        # wrong trace (null if none passed verification)

Pairs without a verified chosen trace are dropped.
"""

import argparse
import json
import os

from data.prompts import (EMOTIONS, answer_from_preference, build_question,
                          flip_answer, load_pairs, pair_key)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--emotion_types", nargs="+", default=list(EMOTIONS), choices=EMOTIONS)
    parser.add_argument("--pairs_dir", default="pairs/msp_podcast")
    parser.add_argument("--verified_dir", default="outputs/reasoning/verified", help="Contains <mode>/<emotion>.json")
    parser.add_argument("--output_dir", default="outputs/traces")
    return parser.parse_args()


def load_verified(verified_dir, mode, emotion):
    path = os.path.join(verified_dir, mode, f"{emotion}.json")
    if not os.path.exists(path):
        print(f"[warning] {path} not found")
        return {}
    with open(path) as f:
        return json.load(f)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for emotion in args.emotion_types:
        blind = load_verified(args.verified_dir, "blind", emotion)
        guided = load_verified(args.verified_dir, "guided", emotion)
        wrong = load_verified(args.verified_dir, "wrong", emotion)

        rows = load_pairs(os.path.join(args.pairs_dir, f"{emotion}_preference.csv"), split="Train")
        records = []
        for row in rows:
            key = pair_key(row["sen1"], row["sen2"])
            if key in blind:
                chosen, source = blind[key], "blind"
            elif key in guided:
                chosen, source = guided[key], "guided"
            else:
                continue

            answer = answer_from_preference(row["preference"])
            rejected = wrong.get(key)
            records.append(
                {
                    "key": key,
                    "emotion": emotion,
                    "sen1": row["sen1"],
                    "sen2": row["sen2"],
                    "preference": int(row["preference"]),
                    "question": build_question(emotion),
                    "answer": answer,
                    "reasoning": chosen["reasoning"],
                    "reasoning_source": source,
                    "rejected_answer": rejected["answer"] if rejected else flip_answer(answer),
                    "rejected_reasoning": rejected["reasoning"] if rejected else None,
                }
            )

        output = os.path.join(args.output_dir, f"{emotion}.jsonl")
        with open(output, "w") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        n_blind = sum(r["reasoning_source"] == "blind" for r in records)
        n_rejected = sum(r["rejected_reasoning"] is not None for r in records)
        print(
            f"{emotion}: {len(records)}/{len(rows)} pairs with a chosen trace "
            f"(blind {n_blind}, guided {len(records) - n_blind}), {n_rejected} with a rejected trace -> {output}"
        )


if __name__ == "__main__":
    main()
