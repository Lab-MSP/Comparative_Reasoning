# Data Generation Pipeline

This pipeline produces the reasoning traces used by SFT-CoT and DPO-CoT. The final traces are
released on [Hugging Face](https://huggingface.co/datasets/Lab-MSP/comparative-reasoning-traces). Run every command from the repository root.

Only the MSP-Podcast training pairs need traces, so steps 1–2 process the 49,953 training audios
listed in `pairs/msp_podcast/train_audio_list.txt`.

## 1. Audio captions

Serve the captioner with vLLM, then caption every training audio:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Captioner --allowed-local-media-path /path/to/msp_podcast \
  --port 8001 --host 127.0.0.1 --dtype bfloat16 --max-model-len 32768 -tp 2

# In another terminal
python -m data.preprocess.qwen_omni_captioner --audio_dir /path/to/msp_podcast/Audios   # -> outputs/captions
```

## 2. GeMAPS-36 acoustic features

```bash
# One 36-dim vector per audio (18 GeMAPS descriptors x mean / normalized std)  -> outputs/gemaps36
python -m data.preprocess.precompute_gemaps36 --audios_dir /path/to/msp_podcast/Audios

# z-normalize each feature over the training audios and map it to low / medium / high  -> outputs/gemaps36_text
python -m data.preprocess.open_smile_to_text
```


## 3. Reasoning traces

Serve the reasoning LLM with vLLM:

```bash
vllm serve Qwen/Qwen3-Next-80B-A3B-Thinking-FP8 --max_model_len 16384 --host 127.0.0.1 -tp 2 \
  --reasoning-parser deepseek_r1 --port 8000
```

Traces are generated in three modes (prompts in `reasoning/prompts/`):

| Mode | Prompt | Answer given to the LLM | Used as |
|---|---|---|---|
| `blind` | `blind.txt` | none | chosen response (primary) |
| `guided` | `guided.txt` | ground truth | chosen response for pairs where `blind` is wrong |
| `wrong` | `guided.txt` | flipped (wrong) answer | rejected response in DPO-CoT |

```bash
for EMOTION in arousal valence dominance; do
  for MODE in blind guided wrong; do
    python -m data.reasoning.generate_reasoning -e $EMOTION -m $MODE   # -> outputs/reasoning/<mode>/<emotion>/*.txt
    python -m data.reasoning.verify_reasoning   -e $EMOTION -m $MODE   # -> outputs/reasoning/verified/<mode>/<emotion>.json
  done
done

# Merge into one trace file per attribute (the released format)  -> outputs/traces/<emotion>.jsonl
python -m data.reasoning.build_traces
```

`generate_reasoning.py` skips pairs whose trace already exists, so re-run it to retry failed requests.
`verify_reasoning.py` keeps a trace only if its final answer matches the ground truth (`blind`,
`guided`) or the flipped answer (`wrong`). `build_traces.py` uses the `blind` trace when it passed
verification and the `guided` trace otherwise.


## 4. Convert to ms-swift format

```bash
AUDIO=/path/to/msp_podcast/Audios

# SFT
python -m data.post_process.make_swift_data -f sft -a $AUDIO -o outputs/swift/sft.jsonl

# SFT-CoT
python -m data.post_process.make_swift_data -f sft -a $AUDIO -o outputs/swift/sft_cot.jsonl -t outputs/traces

# DPO (rejected: the other clip)
python -m data.post_process.make_swift_data -f dpo -a $AUDIO -o outputs/swift/dpo.jsonl

# DPO-CoT (rejected: wrong trace + the other clip)
python -m data.post_process.make_swift_data -f dpo -a $AUDIO -o outputs/swift/dpo_cot.jsonl -t outputs/traces
```

`-t` also accepts the traces downloaded from [Hugging Face](https://huggingface.co/datasets/Lab-MSP/comparative-reasoning-traces) (`data/*.jsonl`). In CoT mode, pairs without a verified trace are skipped. For the cross-emotion experiment, add `-i pairs/msp_podcast/arousal_preference.csv` to train on arousal only.
