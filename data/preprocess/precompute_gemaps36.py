from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from data.preprocess.gemaps36 import GeMAPS36Extractor, gemaps36_feature_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--audios_dir",
        type=str,
        default="/path/to/msp_podcast/Audios",
        help="Path to Audios/ folder containing wav files",
    )
    ap.add_argument(
        "--list_file",
        type=str,
        default="pairs/msp_podcast/train_audio_list.txt",
        help="Audio names (without .wav) to process, one per line. Empty: all .wav files in --audios_dir",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="outputs/gemaps36",
        help="Directory to write one .npy vector per audio",
    )
    ap.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing .npy outputs"
    )
    args = ap.parse_args()

    audios_dir = Path(args.audios_dir)
    out_dir = Path(args.out_dir)

    if not audios_dir.exists():
        raise FileNotFoundError(f"Missing audios_dir: {audios_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the feature names once (read by open_smile_to_text.py)
    names_path = out_dir / "gemaps36_names.txt"
    if not names_path.exists() or args.overwrite:
        names = gemaps36_feature_names()
        names_path.write_text("\n".join(names) + "\n", encoding="utf-8")

    extractor = GeMAPS36Extractor()

    if args.list_file:
        # Read the list of audio filenames to process
        with open(args.list_file, "r") as f:
            audio_filenames = set(line.strip() for line in f if line.strip())
        print(f"Processing {len(audio_filenames)} audios from list: {args.list_file}")
        wavs = [audios_dir / (fname + ".wav") for fname in audio_filenames]
    else:
        wavs = sorted(audios_dir.glob("*.wav"))
    if not wavs:
        print(f"No .wav files found in {audios_dir}")
        return

    print(f"Found {len(wavs)} wav files in: {audios_dir}")
    print(f"Writing .npy vectors to: {out_dir}")

    for wav_path in tqdm(wavs, desc="Extracting GeMAPS36"):
        out_path = out_dir / f"{wav_path.stem}.npy"
        if out_path.exists() and not args.overwrite:
            continue
        try:
            vec = extractor.extract_vector(wav_path)  # (36,)
            # Save as float32
            np.save(out_path, vec.astype(np.float32), allow_pickle=False)
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")

    print(f"\nDone. Example output file: {out_dir / (wavs[0].stem + '.npy')}")
    print(f"Feature names saved to: {names_path}")


if __name__ == "__main__":
    main()
