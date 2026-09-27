from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

# -------------------------
# Feature grouping map: disentangled GeMAPS semantic groups
# -------------------------

DEFAULT_GROUPS: Dict[str, List[str]] = {
    # =====================
    # Prosody / Pitch
    # =====================
    "Pitch level (F0)": ["F0semitoneFrom27.5Hz_sma3nz_amean"],
    "Pitch variability (F0)": ["F0semitoneFrom27.5Hz_sma3nz_stddevNorm"],
    "Jitter level (frequency instability)": ["jitterLocal_sma3nz_amean"],
    "Jitter variability (frequency instability)": ["jitterLocal_sma3nz_stddevNorm"],
    "Shimmer level (amplitude instability)": ["shimmerLocaldB_sma3nz_amean"],
    "Shimmer variability (amplitude instability)": ["shimmerLocaldB_sma3nz_stddevNorm"],
    # =====================
    # Loudness / Energy
    # =====================
    "Loudness level": ["loudness_sma3_amean"],
    "Loudness variability": ["loudness_sma3_stddevNorm"],
    # =====================
    # Voice Quality / Harmonics
    # =====================
    "Harmonicity level (HNR)": ["HNRdBACF_sma3nz_amean"],
    "Harmonicity variability (HNR)": ["HNRdBACF_sma3nz_stddevNorm"],
    # =====================
    # Spectral Balance
    # =====================
    "Spectral tilt level (alphaRatio)": ["alphaRatioV_sma3nz_amean"],
    "Spectral tilt variability (alphaRatio)": ["alphaRatioV_sma3nz_stddevNorm"],
    "Spectral prominence level (Hammarberg index)": ["hammarbergIndexV_sma3nz_amean"],
    "Spectral prominence variability (Hammarberg index)": [
        "hammarbergIndexV_sma3nz_stddevNorm"
    ],
    "Spectral slope level (0–500 Hz)": ["slopeV0-500_sma3nz_amean"],
    "Spectral slope variability (0–500 Hz)": ["slopeV0-500_sma3nz_stddevNorm"],
    "Spectral slope level (500–1500 Hz)": ["slopeV500-1500_sma3nz_amean"],
    "Spectral slope variability (500–1500 Hz)": ["slopeV500-1500_sma3nz_stddevNorm"],
    # =====================
    # Formants (Articulation)
    # =====================
    "Formant F1 level (frequency)": ["F1frequency_sma3nz_amean"],
    "Formant F1 variability (frequency)": ["F1frequency_sma3nz_stddevNorm"],
    "Formant F1 level (bandwidth)": ["F1bandwidth_sma3nz_amean"],
    "Formant F1 variability (bandwidth)": ["F1bandwidth_sma3nz_stddevNorm"],
    "Formant F2 level (frequency)": ["F2frequency_sma3nz_amean"],
    "Formant F2 variability (frequency)": ["F2frequency_sma3nz_stddevNorm"],
    "Formant F3 level (frequency)": ["F3frequency_sma3nz_amean"],
    "Formant F3 variability (frequency)": ["F3frequency_sma3nz_stddevNorm"],
    # =====================
    # Harmonic Differences
    # =====================
    "H1-H2 level (relative)": ["logRelF0-H1-H2_sma3nz_amean"],
    "H1-H2 variability (relative)": ["logRelF0-H1-H2_sma3nz_stddevNorm"],
    "H1-A3 level (relative)": ["logRelF0-H1-A3_sma3nz_amean"],
    "H1-A3 variability (relative)": ["logRelF0-H1-A3_sma3nz_stddevNorm"],
    # =====================
    # Formant Amplitudes relative to F0
    # =====================
    "F1 amplitude rel. F0 level": ["F1amplitudeLogRelF0_sma3nz_amean"],
    "F1 amplitude rel. F0 variability": ["F1amplitudeLogRelF0_sma3nz_stddevNorm"],
    "F2 amplitude rel. F0 level": ["F2amplitudeLogRelF0_sma3nz_amean"],
    "F2 amplitude rel. F0 variability": ["F2amplitudeLogRelF0_sma3nz_stddevNorm"],
    "F3 amplitude rel. F0 level": ["F3amplitudeLogRelF0_sma3nz_amean"],
    "F3 amplitude rel. F0 variability": ["F3amplitudeLogRelF0_sma3nz_stddevNorm"],
}


@dataclass
class FeatureStats:
    """Holds normalization stats per feature name."""

    mean: Dict[str, float]
    std: Dict[str, float]


def _zscore(value: float, mean: float, std: float, eps: float = 1e-8) -> float:
    return (value - mean) / (std + eps)


def _bin_label(z: float, bins: int = 3) -> str:
    """
    Convert z-score to categorical label.
    bins=3: low / medium / high
    bins=5: very low / low / medium / high / very high
    """
    if not np.isfinite(z):
        return "unknown"
    if bins == 3:
        # thresholds roughly correspond to bottom/middle/top terciles for near-normal data
        if z <= -0.43:
            return "low"
        elif z >= 0.43:
            return "high"
        else:
            return "medium"

    if bins == 5:
        if z <= -0.84:
            return "very low"
        elif z <= -0.25:
            return "low"
        elif z < 0.25:
            return "medium"
        elif z < 0.84:
            return "high"
        else:
            return "very high"

    raise ValueError("bins must be 3 or 5")


def summarize_opensmile_semantic(
    vector: np.ndarray,
    feature_names: List[str],
    stats: FeatureStats,
    groups: Dict[str, List[str]] = DEFAULT_GROUPS,
    bins: int = 3,
    include_z: bool = False,
) -> str:
    """
    Create a semantic summary string from openSMILE features.
    - Uses z-score normalization based on dataset stats.
    - Aggregates multiple features in a group by averaging z-scores.
    - Returns a multi-line text suitable for prompt insertion.
    """
    name_to_val = {n: float(v) for n, v in zip(feature_names, vector.tolist())}

    lines: List[str] = []
    for group_name, feats in groups.items():
        zs = []
        for f in feats:
            if f not in name_to_val:
                continue
            if f not in stats.mean or f not in stats.std:
                continue
            zs.append(_zscore(name_to_val[f], stats.mean[f], stats.std[f]))

        if not zs:
            continue

        z_avg = float(np.nanmean(zs))
        label = _bin_label(z_avg, bins=bins)

        if include_z:
            if np.isfinite(z_avg):
                lines.append(f"- {group_name}: {label} (z={z_avg:+.2f})")
            else:
                lines.append(f"- {group_name}: {label}")
        else:
            lines.append(f"- {group_name}: {label}")

    return "\n".join(lines)


def compute_feature_stats(
    all_vectors: np.ndarray, feature_names: List[str]
) -> FeatureStats:
    """
    all_vectors: shape (N, D)
    """
    vectors = np.array(all_vectors, dtype=np.float64, copy=True)
    vectors[~np.isfinite(vectors)] = np.nan
    means = np.nanmean(vectors, axis=0)
    stds = np.nanstd(vectors, axis=0)

    return FeatureStats(
        mean={n: float(m) for n, m in zip(feature_names, means.tolist())},
        std={n: float(s) for n, s in zip(feature_names, stds.tolist())},
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert openSMILE vectors to semantic text summaries."
    )
    parser.add_argument(
        "--features_dir",
        type=str,
        default="outputs/gemaps36",
        help="Path to directory containing .npy feature vectors",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="outputs/gemaps36_text",
        help="Path to directory to save semantic text summaries",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=3,
        choices=[3, 5],
        help="Number of levels for the z-scores (3: low/medium/high, used in the paper)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    with open(
        Path(args.features_dir) / "gemaps36_names.txt", "r", encoding="utf-8"
    ) as f:
        feature_names = [line.strip() for line in f if line.strip()]

    npy_paths = sorted(Path(args.features_dir).glob("*.npy"))
    if not npy_paths:
        print(f"No .npy files found in {args.features_dir}")
        return

    print(f"Found {len(npy_paths)} .npy files in: {args.features_dir}")
    print("Loading feature vectors to compute dataset stats...")
    all_vectors = []
    for npy_path in tqdm(npy_paths, desc="Loading features", dynamic_ncols=True):
        vec = np.load(npy_path)
        if vec.ndim > 1:
            vec = vec.reshape(-1)
        all_vectors.append(vec)
    all_vectors = np.stack(all_vectors, axis=0)  # (N, D)

    if all_vectors.ndim != 2:
        raise ValueError(f"Expected (N, D) array, got shape: {all_vectors.shape}")

    n_samples, dim = all_vectors.shape
    if dim != 36:
        raise ValueError(f"Expected D=36 features, got D={dim}")

    if len(feature_names) != dim:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) must match D ({dim})"
        )

    stats = compute_feature_stats(all_vectors, feature_names)
    print(f"Computed z-score statistics over {n_samples} audios")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Generating semantic summaries...")
    for npy_path in tqdm(npy_paths, desc="Summarizing", dynamic_ncols=True):
        vec = np.load(npy_path)
        if vec.ndim > 1:
            vec = vec.reshape(-1)
        if vec.shape[0] != 36:
            raise ValueError(
                f"Invalid feature length for {npy_path.name}: {vec.shape[0]}"
            )

        summary = summarize_opensmile_semantic(
            vector=vec,
            feature_names=feature_names,
            stats=stats,
            bins=args.bins,
        )

        out_path = save_dir / f"{npy_path.stem}.txt"
        out_path.write_text(summary + "\n", encoding="utf-8")

    print(f"Saved {len(npy_paths)} summaries to: {save_dir}")


if __name__ == "__main__":
    main()
