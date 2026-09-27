"""Caption every audio file with Qwen3-Omni-Captioner served by vLLM.

Each caption is saved to <output_dir>/<audio>.txt; existing files are skipped, so the script
can be re-run to resume. The vLLM server must be started with
--allowed-local-media-path covering --audio_dir.
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio_dir", default="/path/to/msp_podcast/Audios")
    parser.add_argument(
        "--list_file",
        default="pairs/msp_podcast/train_audio_list.txt",
        help="Audio names (without .wav) to caption, one per line. Empty: all .wav files in --audio_dir",
    )
    parser.add_argument("--output_dir", default="outputs/captions")
    parser.add_argument("--num_workers", "-w", type=int, default=256)
    parser.add_argument("--vllm_base_url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--vllm_model", default="Qwen/Qwen3-Omni-30B-A3B-Captioner")
    return parser.parse_args()


def list_audios(audio_dir, list_file):
    if list_file:
        with open(list_file) as f:
            return [Path(audio_dir) / f"{line.strip()}.wav" for line in f if line.strip()]
    return sorted(Path(audio_dir).glob("*.wav"))


def caption(client, model, audio_path: Path):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [{"type": "audio_url", "audio_url": {"url": f"file://{os.path.abspath(audio_path)}"}}],
            }
        ],
        temperature=0.6,
        top_p=0.95,
        max_tokens=2048,
        extra_body={"top_k": 20},
    )
    return completion.choices[0].message.content.strip()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    audios = list_audios(args.audio_dir, args.list_file)
    tasks = [a for a in audios if not (Path(args.output_dir) / f"{a.stem}.txt").exists()]
    print(f"{len(audios)} audios, {len(audios) - len(tasks)} already captioned, {len(tasks)} to caption")
    if not tasks:
        return

    client = OpenAI(base_url=args.vllm_base_url, api_key="EMPTY")

    def run(audio_path):
        (Path(args.output_dir) / f"{audio_path.stem}.txt").write_text(caption(client, args.vllm_model, audio_path))

    errors = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {pool.submit(run, audio): audio for audio in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Captioning"):
            if future.exception() is not None:
                errors.append(f"{futures[future].name}: {future.exception()}")

    print(f"Captioned {len(tasks) - len(errors)} audios to {args.output_dir}, {len(errors)} errors")
    if errors:
        error_file = os.path.join(args.output_dir, "..", "caption_errors.txt")
        Path(error_file).write_text("\n".join(errors))
        print(f"Errors saved to {error_file} (re-run the script to retry)")


if __name__ == "__main__":
    main()
