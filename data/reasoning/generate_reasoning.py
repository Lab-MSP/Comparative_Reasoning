"""Generate comparative reasoning traces with a reasoning LLM served by vLLM.

Modes:
  blind  : the LLM decides the answer itself (primary traces for SFT-CoT / DPO-CoT chosen responses)
  guided : the ground-truth answer is given (backup traces for pairs the blind mode got wrong)
  wrong  : the flipped (wrong) answer is given (rejected responses for DPO-CoT)

Each trace is saved to <output_dir>/<sen1>_<sen2>.txt; existing files are skipped, so the
script can be re-run to resume.
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from data.prompts import (
    EMOTIONS,
    REASONING_GUIDELINES,
    answer_from_preference,
    build_question,
    flip_answer,
    load_pairs,
    pair_key,
)

PROMPT_DIR = Path(__file__).parent / "prompts"
PROMPT_FILES = {"blind": "blind.txt", "guided": "guided.txt", "wrong": "guided.txt"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--emotion_type", "-e", required=True, choices=EMOTIONS)
    parser.add_argument("--mode", "-m", default="blind", choices=list(PROMPT_FILES))
    parser.add_argument("--pairs_dir", default="pairs/msp_podcast", help="Directory with <emotion>_preference.csv")
    parser.add_argument("--caption_dir", default="outputs/captions", help="Audio captions (<audio>.txt)")
    parser.add_argument("--gemaps_text_dir", default="outputs/gemaps36_text", help="GeMAPS text levels (<audio>.txt)")
    parser.add_argument("--output_dir", default=None, help="Default: outputs/reasoning/<mode>/<emotion_type>")
    parser.add_argument("--num_workers", "-w", type=int, default=256)
    parser.add_argument("--vllm_base_url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--vllm_model", default="Qwen/Qwen3-Next-80B-A3B-Thinking-FP8")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = f"outputs/reasoning/{args.mode}/{args.emotion_type}"
    return args


def build_user_prompt(template, emotion, caption_1, caption_2, gemaps_1, gemaps_2, given_answer):
    return (
        template.replace("__AUDIO_DESCRIPTION_1__", caption_1)
        .replace("__AUDIO_DESCRIPTION_2__", caption_2)
        .replace("__OPENSMILE_FEATURES_1__", gemaps_1)
        .replace("__OPENSMILE_FEATURES_2__", gemaps_2)
        .replace("__EMOTION_TYPE__", emotion)
        .replace("__QUESTION__", build_question(emotion) + " ")
        .replace("__GROUND_TRUTH_ANSWER__", given_answer)
        .replace("__REASONING_GUIDELINE__", REASONING_GUIDELINES[emotion])
    )


def generate(client, model, user_prompt):
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [{"type": "text", "text": user_prompt}]}],
        temperature=0.6,
        top_p=0.95,
        max_tokens=8192,
        extra_body={"top_k": 20, "min_p": 0},
    )
    choice = completion.choices[0]
    if choice.finish_reason == "length":
        raise ValueError("length exceeded")
    response = (choice.message.content or "").strip()
    if not response:
        raise ValueError("empty response")
    return response


def main():
    args = parse_args()
    emotion = args.emotion_type
    template = (PROMPT_DIR / PROMPT_FILES[args.mode]).read_text()
    os.makedirs(args.output_dir, exist_ok=True)

    def read(directory, audio):
        return (Path(directory) / f"{Path(audio).stem}.txt").read_text()

    tasks = []
    rows = load_pairs(os.path.join(args.pairs_dir, f"{emotion}_preference.csv"), split="Train")
    for row in rows:
        key = pair_key(row["sen1"], row["sen2"])
        if os.path.exists(os.path.join(args.output_dir, f"{key}.txt")):
            continue
        answer = answer_from_preference(row["preference"])
        given_answer = flip_answer(answer) if args.mode == "wrong" else answer
        user_prompt = build_user_prompt(
            template,
            emotion,
            read(args.caption_dir, row["sen1"]),
            read(args.caption_dir, row["sen2"]),
            read(args.gemaps_text_dir, row["sen1"]),
            read(args.gemaps_text_dir, row["sen2"]),
            given_answer,
        )
        tasks.append((key, user_prompt))

    print(f"{len(rows)} train pairs, {len(rows) - len(tasks)} already done, {len(tasks)} to generate")
    if not tasks:
        return

    client = OpenAI(base_url=args.vllm_base_url, api_key="EMPTY")

    def run(key, user_prompt):
        response = generate(client, args.vllm_model, user_prompt)
        Path(args.output_dir, f"{key}.txt").write_text(response)

    errors = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {pool.submit(run, key, prompt): key for key, prompt in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{args.mode}/{emotion}"):
            if future.exception() is not None:
                errors.append(f"{futures[future]}: {future.exception()}")

    print(f"Generated {len(tasks) - len(errors)} traces to {args.output_dir}, {len(errors)} errors")
    if errors:
        error_file = os.path.join(args.output_dir, "..", f"{emotion}_errors.txt")
        Path(error_file).write_text("\n".join(errors))
        print(f"Errors saved to {error_file} (re-run the script to retry)")


if __name__ == "__main__":
    main()
