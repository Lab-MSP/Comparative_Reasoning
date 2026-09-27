"""Evaluate an audio LM on preference pairs with vLLM (greedy decoding).

Trained checkpoints are asked the same question as in training. With --zero_shot, the base
model is additionally instructed to answer in the <answer> Clip 1/2 </answer> format.
"""

import argparse
import json
import os
import re
import time

# vLLM settings for reproducible results; set before importing vLLM
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"

from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniProcessor
from vllm import LLM, SamplingParams

from data.prompts import EMOTIONS, SYSTEM_PROMPT, build_question, emotion_from_path, load_pairs, pair_key

ZERO_SHOT_FORMAT = " Answer in the format: <answer> Clip 1 </answer> or <answer> Clip 2 </answer>."


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-Omni-3B", help="Merged checkpoint dir or HF model id")
    parser.add_argument("--data_files", nargs="+", required=True, help="<emotion>_preference.csv files")
    parser.add_argument("--out_files", nargs="+", required=True, help="One output JSON per data file")
    parser.add_argument("--audio_dir", required=True, help="Directory containing the audio files")
    parser.add_argument("--split", default="Test1", help="Split to evaluate (empty string: all rows)")
    parser.add_argument("--zero_shot", action="store_true", help="Add the answer-format instruction")
    args = parser.parse_args()
    if len(args.data_files) != len(args.out_files):
        parser.error("--data_files and --out_files must have the same length")
    return args


def build_conversation(row, emotion, audio_dir, zero_shot):
    question = build_question(emotion)
    if zero_shot:
        question += ZERO_SHOT_FORMAT
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": os.path.join(audio_dir, row["sen1"])},
                {"type": "audio", "audio": os.path.join(audio_dir, row["sen2"])},
                {"type": "text", "text": question},
            ],
        },
    ]


def parse_prediction(response: str):
    """Return 1 if the model picks Clip 1, 0 for Clip 2, None if unparsable."""
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)
    answer = (match.group(1).strip() if match and match.group(1).strip() else response).lower()

    if answer in {"clip 1", "clip1", "1", "first", "1st"}:
        return 1
    if answer in {"clip 2", "clip2", "2", "second", "2nd"}:
        return 0
    if re.search(r"\bclip\s*1\b", answer):
        return 1
    if re.search(r"\bclip\s*2\b", answer):
        return 0
    return None


def evaluate(llm, processor, sampling_params, rows, emotion, audio_dir, zero_shot):
    inputs = []
    for row in rows:
        conversation = build_conversation(row, emotion, audio_dir, zero_shot)
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, _, _ = process_mm_info(conversation, use_audio_in_video=False)
        inputs.append({"prompt": prompt, "multi_modal_data": {"audio": audios}})

    outputs = llm.generate(inputs, sampling_params=sampling_params)

    results = []
    for row, output in zip(rows, outputs):
        response = output.outputs[0].text.strip()
        prediction = parse_prediction(response)
        preference = int(row["preference"])
        results.append(
            {
                "key": pair_key(row["sen1"], row["sen2"]),
                "sen1": row["sen1"],
                "sen2": row["sen2"],
                "preference": preference,
                "prediction": prediction,
                "correct": prediction == preference,
                "response": response,
            }
        )
    return results


def main():
    args = parse_args()
    start = time.time()

    # The processor is only used for the chat template; vLLM handles tokenization and inference.
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    llm = LLM(
        model=args.model_name,
        max_model_len=8192,
        trust_remote_code=True,
        dtype="bfloat16",
        enable_prefix_caching=False,
        enforce_eager=True,
        limit_mm_per_prompt={"audio": 2, "image": 1, "video": 1},
    )
    sampling_params = SamplingParams(max_tokens=4096, temperature=0, top_k=-1, seed=1234)

    for data_file, out_file in zip(args.data_files, args.out_files):
        emotion = emotion_from_path(data_file)
        assert emotion in EMOTIONS
        rows = load_pairs(data_file, split=args.split or None)
        results = evaluate(llm, processor, sampling_params, rows, emotion, args.audio_dir, args.zero_shot)

        correct = sum(r["correct"] for r in results)
        invalid = sum(r["prediction"] is None for r in results)
        accuracy = correct / len(results) if results else 0.0
        print(f"{emotion}: accuracy {accuracy:.4f} (correct={correct}, invalid={invalid}, total={len(results)})")

        os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
        with open(out_file, "w") as f:
            json.dump(
                {
                    "attribute": emotion,
                    "total": len(results),
                    "correct": correct,
                    "invalid": invalid,
                    "accuracy": accuracy,
                    "results": results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    print(f"Done in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
