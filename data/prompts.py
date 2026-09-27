"""Prompts and pair utilities shared by data generation, training data conversion, and evaluation.

The question built by `build_question` must stay identical between the ms-swift training
data (data/post_process) and evaluation (src/eval_vllm.py).
"""

import csv
from pathlib import Path

EMOTIONS = ("arousal", "valence", "dominance")

SYSTEM_PROMPT = "You are a helpful assistant for emotion comparative reasoning."

ATTRIBUTE_DEFS = {
    "arousal": (
        "Arousal describes the energy and activation level in speech. "
        "High arousal sounds excited, intense, urgent, loud, fast, or agitated. "
        "Low arousal sounds calm, relaxed, quiet, slow, or sleepy."
    ),
    "valence": (
        "Valence describes how positive or negative the emotion sounds. "
        "High valence sounds happy, pleasant, satisfied, or friendly. "
        "Low valence sounds sad, angry, unpleasant, or frustrated."
    ),
    "dominance": (
        "Dominance describes confidence and control in the voice. "
        "High dominance sounds confident, assertive, in-control. "
        "Low dominance sounds uncertain, hesitant, submissive, or powerless."
    ),
}

# Attribute-specific guidelines given to the reasoning LLM (data/reasoning).
REASONING_GUIDELINES = {
    "arousal": (
        "Estimate arousal by determining how energized or activated the speaker appears overall. "
        "Consider how vocal delivery (e.g., loudness, pitch dynamics, speech rate, vocal effort), "
        "temporal intensity changes, and contextual urgency jointly contribute to the perceived "
        "level of activation. Integrate both acoustic and linguistic evidence to infer whether "
        "the speech conveys heightened energy or a subdued state."
    ),
    "valence": (
        "Estimate valence by determining the speaker’s emotional attitude or affective stance. "
        "Integrate semantic meaning, communicative intent, and affective vocal qualities such as "
        "warmth, tension, friendliness, irritation, or amusement. Evaluate how wording, tone, "
        "and delivery together convey overall positivity or negativity, even when surface cues "
        "such as loudness or calmness alone may be ambiguous."
    ),
    "dominance": (
        "Estimate dominance by determining how much confidence or control the speaker conveys "
        "through the interaction. Consider vocal stability, firmness, articulation clarity, "
        "pacing regulation, intensity control, hesitation patterns, and linguistic certainty. "
        "Integrate delivery style and language use to assess whether the speaker appears "
        "assertive and in control or uncertain and yielding."
    ),
}


def build_question(emotion: str) -> str:
    """Question asked to the audio LM (training and evaluation)."""
    return (
        "You will hear two audio clips. Clip 1 is the first audio clip. "
        "Clip 2 is the second audio clip. "
        f"{ATTRIBUTE_DEFS[emotion]} Which clip has higher {emotion}?"
    )


def answer_from_preference(preference) -> str:
    """preference == 1 means the first clip (sen1) has the higher attribute value."""
    return "Clip 1" if str(preference) == "1" else "Clip 2"


def flip_answer(answer: str) -> str:
    return "Clip 2" if answer == "Clip 1" else "Clip 1"


def pair_key(sen1: str, sen2: str) -> str:
    return f"{Path(sen1).stem}_{Path(sen2).stem}"


def emotion_from_path(path: str) -> str:
    """`pairs/msp_podcast/arousal_preference.csv` -> `arousal`."""
    stem = Path(path).stem
    emotion = stem.split("_")[0]
    if emotion not in EMOTIONS:
        raise ValueError(f"Cannot infer the emotion attribute from {path}")
    return emotion


def load_pairs(csv_path: str, split: str | None = "Train") -> list[dict]:
    """Load preference pairs (columns: sen1, sen2, preference, split)."""
    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))
    if split:
        rows = [row for row in rows if row["split"] == split]
    return rows
