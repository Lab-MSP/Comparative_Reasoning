#!/bin/bash
# Evaluate one or more models on a preference benchmark. Run from the repository root.
#
#   bash src/eval.sh <benchmark> <model> [<model> ...]
#
#   benchmark : msp_valid   MSP-Podcast Development subset (1k pairs / attribute), for checkpoint selection
#               msp_test    MSP-Podcast Test1 (3k pairs / attribute), main table
#               biic        BIIC-Podcast (1k pairs / attribute), cross-domain
#               whiser      WHiSER (1k pairs / attribute), cross-domain
#   model     : a LoRA checkpoint dir from src/sft_swift.sh or src/dpo_swift.sh (merged with
#               `swift export` on first use), a merged model dir, or a HF model id.
#               The base model (Qwen/Qwen2.5-Omni-3B) is evaluated zero-shot.
#
# Examples:
#   bash src/eval.sh msp_valid outputs/experiments/<run_name>/checkpoint-*
#   bash src/eval.sh msp_test outputs/experiments/<run_name>/checkpoint-<step>
#   bash src/eval.sh biic Qwen/Qwen2.5-Omni-3B
#
# Per-attribute results: outputs/results/<benchmark>/<model>/<attribute>_eval.json
# Summary:               outputs/results/<benchmark>_summary.csv

set -e

MSP_AUDIO_DIR="/path/to/msp_podcast/Audios"
BIIC_AUDIO_DIR="/path/to/biic_podcast/Audios"
WHISER_AUDIO_DIR="/path/to/whiser/Audios"
PRETRAINED_MODEL="Qwen/Qwen2.5-Omni-3B"

if [ $# -lt 2 ]; then
    sed -n '2,19p' "$0"
    exit 1
fi
BENCHMARK=$1
shift

case $BENCHMARK in
    msp_valid) PAIRS=pairs/msp_podcast;  SUFFIX=mini_valid; SPLIT=Development; AUDIO_DIR=$MSP_AUDIO_DIR ;;
    msp_test)  PAIRS=pairs/msp_podcast;  SUFFIX=preference; SPLIT=Test1;       AUDIO_DIR=$MSP_AUDIO_DIR ;;
    biic)      PAIRS=pairs/biic_podcast; SUFFIX=preference; SPLIT=Test;        AUDIO_DIR=$BIIC_AUDIO_DIR ;;
    whiser)    PAIRS=pairs/whiser;       SUFFIX=preference; SPLIT=Test1;       AUDIO_DIR=$WHISER_AUDIO_DIR ;;
    *) echo "Unknown benchmark: $BENCHMARK"; exit 1 ;;
esac

RESULTS_DIR="outputs/results/${BENCHMARK}"

for MODEL in "$@"; do
    MODEL=${MODEL%/}
    # Skip merged copies picked up by a glob such as checkpoint-* (their adapter is evaluated instead)
    if [[ "$MODEL" == *_merged ]] && [ -f "${MODEL%_merged}/adapter_config.json" ]; then
        continue
    fi
    EXTRA_ARGS=""
    if [ "$MODEL" == "$PRETRAINED_MODEL" ]; then
        NAME="zero_shot"
        EXTRA_ARGS="--zero_shot"
    else
        # e.g. outputs/experiments/<run_name>/checkpoint-<step> -> <run_name>/checkpoint-<step>
        NAME="$(basename "$(dirname "$MODEL")")/$(basename "$MODEL")"
    fi

    # LoRA adapter -> merge into the base model once
    if [ -f "$MODEL/adapter_config.json" ]; then
        MERGED="${MODEL}_merged"
        if [ ! -d "$MERGED" ]; then
            swift export --adapters "$MODEL" --merge_lora true --model "$PRETRAINED_MODEL" --output_dir "$MERGED"
        fi
        MODEL=$MERGED
    fi

    echo "Evaluating ${NAME} on ${BENCHMARK}"
    python -m src.eval_vllm \
        --model_name "$MODEL" \
        --data_files ${PAIRS}/{arousal,dominance,valence}_${SUFFIX}.csv \
        --out_files ${RESULTS_DIR}/${NAME}/{arousal,dominance,valence}_eval.json \
        --split "$SPLIT" \
        --audio_dir "$AUDIO_DIR" \
        $EXTRA_ARGS
done

python -m src.summarize --input_dir "$RESULTS_DIR" --output_csv "${RESULTS_DIR}_summary.csv"
echo "Summary saved to ${RESULTS_DIR}_summary.csv"
