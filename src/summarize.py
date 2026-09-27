import argparse
import csv
import json
from pathlib import Path


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def collect_metrics(input_dir: Path) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for eval_path in input_dir.rglob("*_eval.json"):
        if not eval_path.is_file():
            continue

        attr = eval_path.stem.replace("_eval", "")
        if attr not in {"arousal", "valence", "dominance"}:
            continue

        model_dir = eval_path.parent
        model_key = str(model_dir.relative_to(input_dir))

        try:
            with open(eval_path, "r") as f:
                payload = json.load(f)
        except Exception:
            continue

        acc = payload.get("accuracy")
        if not isinstance(acc, (int, float)):
            continue

        metrics.setdefault(model_key, {})[attr] = float(acc)

    return metrics


def write_csv(metrics: dict[str, dict[str, float]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model",
        "arousal_accuracy",
        "valence_accuracy",
        "dominance_accuracy",
        "average_accuracy",
    ]

    rows = []
    for model_key in sorted(metrics.keys()):
        item = metrics[model_key]
        values = [item.get("arousal"), item.get("valence"), item.get("dominance")]
        present = [v for v in values if v is not None]
        avg = sum(present) / len(present) if present else None

        rows.append(
            {
                "model": model_key,
                "arousal_accuracy": _fmt(item.get("arousal")),
                "valence_accuracy": _fmt(item.get("valence")),
                "dominance_accuracy": _fmt(item.get("dominance")),
                "average_accuracy": _fmt(avg),
            }
        )

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        "-i",
        type=str,
        required=True,
        help="Root directory containing *_eval.json files",
    )
    parser.add_argument(
        "--output_csv",
        "-o",
        type=str,
        required=True,
        help="Output CSV file path",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)

    metrics = collect_metrics(input_dir)
    write_csv(metrics, output_csv)


if __name__ == "__main__":
    main()
