from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import opensmile


# ---- The 18 GeMAPS LLD "base" names (after smoothing), in a fixed order ----
# We will take only two functionals per base:
#   - _amean
#   - _stddevNorm
#
# That gives 18 * 2 = 36 dims.
GEMAPS18_BASES: List[str] = [
    # Frequency related
    "F0semitoneFrom27.5Hz_sma3nz",
    "jitterLocal_sma3nz",
    "F1frequency_sma3nz",
    "F1bandwidth_sma3nz",
    "F2frequency_sma3nz",
    "F3frequency_sma3nz",
    # Energy / amplitude related
    "shimmerLocaldB_sma3nz",
    "loudness_sma3",
    "HNRdBACF_sma3nz",
    # Spectral balance related
    "alphaRatioV_sma3nz",
    "hammarbergIndexV_sma3nz",
    "slopeV0-500_sma3nz",
    "slopeV500-1500_sma3nz",
    # Relative energies / harmonic differences (still part of the 18 LLD list)
    "F1amplitudeLogRelF0_sma3nz",
    "F2amplitudeLogRelF0_sma3nz",
    "F3amplitudeLogRelF0_sma3nz",
    "logRelF0-H1-H2_sma3nz",
    "logRelF0-H1-A3_sma3nz",
]


def gemaps36_feature_names() -> List[str]:
    """
    Returns the 36 dimension names in the exact order used by the extractor.
    """
    names: List[str] = []
    for base in GEMAPS18_BASES:
        names.append(f"{base}_amean")
        names.append(f"{base}_stddevNorm")
    return names


@dataclass
class GeMAPS36:
    values: np.ndarray  # shape (36,)
    names: List[str]    # length 36


class GeMAPS36Extractor:
    """
    Extract ONE fixed-length 36D vector for an entire wav file:
      - GeMAPSv01b
      - FeatureLevel.Functionals
      - Keep ONLY {amean, stddevNorm} for the 18 LLDs (36 dims total)
    """

    def __init__(self, use_fast_processor: bool = True) -> None:
        # openSMILE config via opensmile-python
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.GeMAPSv01b,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        self._names_36 = gemaps36_feature_names()

    def extract(self, wav_path: str | Path) -> GeMAPS36:
        wav_path = Path(wav_path)
        if not wav_path.exists():
            raise FileNotFoundError(f"Missing wav file: {wav_path}")

        df = self.smile.process_file(str(wav_path))  # expect 1 row
        if df.shape[0] != 1:
            raise RuntimeError(f"Expected 1 row for Functionals, got {df.shape[0]} rows: {wav_path}")

        # Validate required columns exist
        missing = [n for n in self._names_36 if n not in df.columns]
        if missing:
            # If this happens, print a few columns to debug.
            sample_cols = list(df.columns)[:30]
            raise RuntimeError(
                "GeMAPS36 columns missing from openSMILE output.\n"
                f"Missing ({len(missing)}): {missing}\n"
                f"Sample columns: {sample_cols}\n"
                "Check opensmile version / feature_set / feature_level."
            )

        # Pull them in the exact order
        values = df.loc[df.index[0], self._names_36].to_numpy(dtype=np.float32)
        if values.shape[0] != 36:
            raise RuntimeError(f"Expected 36 dims, got {values.shape[0]} dims for {wav_path}")

        return GeMAPS36(values=values, names=self._names_36)

    def extract_vector(self, wav_path: str | Path) -> np.ndarray:
        return self.extract(wav_path).values


class SentenceAudio:
    """
    Convenience wrapper so you can do:
      sent1 = SentenceAudio('/path/to/file.wav')
      v = sent1.get_GeMAPS_LLD()
    """

    def __init__(self, wav_path: str | Path, extractor: Optional[GeMAPS36Extractor] = None):
        self.wav_path = Path(wav_path)
        self.extractor = extractor if extractor is not None else GeMAPS36Extractor()

    def get_GeMAPS_LLD(self) -> np.ndarray:
        """
        Returns the 36D GeMAPS LLD functional vector (float32).
        """
        return self.extractor.extract_vector(self.wav_path)

    def get_GeMAPS_LLD_with_names(self) -> Tuple[np.ndarray, List[str]]:
        feat = self.extractor.extract(self.wav_path)
        return feat.values, feat.names