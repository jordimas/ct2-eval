#!/usr/bin/env python3
"""
Unified CTranslate2 Benchmark Tool

Modes: whisper | translation | generation
"""

import argparse
import json
import os
import time

import ctranslate2
import numpy as np


def cv_percent(mean, std):
    return (std / mean * 100) if mean != 0 else 0


def load_whisper_data(args):
    from datasets import load_dataset
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    normalizer_path = os.path.join(os.path.dirname(__file__), "normalizer.json")
    if os.path.exists(normalizer_path):
        with open(normalizer_path) as f:
            normalizer = EnglishTextNormalizer(json.load(f))
    else:
        normalizer = EnglishTextNormalizer({})

    print("Loading LibriSpeech dataset...")
    dataset = load_dataset("librispeech_asr", "clean", split="validation", streaming=True)

    samples = []
    for i, s in enumerate(dataset):
        samples.append({
            "audio": s["audio"]["array"],
            "rate": s["audio"]["sampling_rate"],
            "ref": s["text"],
        })
        if args.num_samples and i + 1 >= args.num_samples:
            break

    audio_duration = sum(len(s["audio"]) / s["rate"] for s in samples)
    return samples, normalizer, audio_duration


def load_translation_data(args):
    import sentencepiece as spm

    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer_path)

    with open(args.source_file) as f:
        sources = [l.strip() for l in f if l.strip()]
    with open(args.target_file) as f:
        targets = [l.strip() for l in f if l.strip()]

    if args.num_samples:
        sources, targets = sources[:args.num_samples], targets[:args.num_samples]

    samples = [{"src": s, "ref": t, "tokens": tokenizer.encode(s, out_type=str)}
               for s, t in zip(sources, targets)]
    total_tokens = sum(len(s["tokens"]) for s in samples)
    return samples, tokenizer, total_tokens


def load_generation_data(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    with open(args.source_file) as f:
        sources = [l.strip() for l in f if l.strip()]
    with open(args.target_file) as f:
        targets = [l.strip() for l in f if l.strip()]

    if args.num_samples:
        sources, targets = sources[:args.num_samples], targets[:args.num_samples]

    samples = [{"src": s, "ref": t} for s, t in zip(sources, targets)]
    total_tokens = sum(len(tokenizer.encode(s["src"])) for s in samples)
    return samples, tokenizer, total_tokens


def run_whisper(model, samples):
    preds = []
    for s in samples:
        segments, _ = model.transcribe(s["audio"], language="en")
        preds.append("".join(seg.text for seg in segments))
    return preds, [s["ref"] for s in samples]


def run_translation(model, samples, tokenizer):
    results = model.translate_batch([s["tokens"] for s in samples])
    preds = [tokenizer.decode(r.hypotheses[0]) for r in results]
    return preds, [s["ref"] for s in samples]


def run_generation(model, samples, tokenizer, args):
    preds = []
    for s in samples:
        prompt = args.prompt_template.format(text=s["src"])
        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt))
        result = model.generate_batch(
            [tokens], max_length=args.max_length,
            sampling_temperature=args.temperature, include_prompt_in_result=False
        )
        preds.append(tokenizer.convert_tokens_to_string(result[0].sequences[0]))
    return preds, [s["ref"] for s in samples]


def compute_wer(preds, refs, normalizer):
    from jiwer import wer
    return 100 * wer([normalizer(p) for p in preds], [normalizer(r) for r in refs])


def compute_bleu(preds, refs):
    from sacrebleu.metrics import BLEU
    return BLEU().corpus_score(preds, [refs]).score


MODELS = {
    "whisper": {"model": "small"},
    "translation": {"model": "softcatala/translate-eng-cat", "tokenizer": "softcatala/translate-eng-cat"},
    "generation": {"model": "gemma-3-270m", "tokenizer": "google/gemma-3-270m"},
}


def main():
    parser = argparse.ArgumentParser(description="CTranslate2 Benchmark")
    parser.add_argument("--mode", required=True, choices=["whisper", "translation", "generation"])
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--warmup_runs", type=int, default=1)
    parser.add_argument("--source_file", default="flores200.eng")
    parser.add_argument("--target_file", default="flores200.cat")
    parser.add_argument("--prompt_template", default="<start_of_turn>user\nTranslate to Catalan:\n{text}<end_of_turn>\n<start_of_turn>model\n")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()
    
    # Set model paths from defaults
    args.model_path = MODELS[args.mode]["model"]
    args.tokenizer_path = MODELS[args.mode].get("tokenizer")

    # Setup devices
    if args.devices == "auto":
        devices = ["cpu"] + (["cuda"] if ctranslate2.get_cuda_device_count() > 0 else [])
    else:
        devices = [d.strip() for d in args.devices.split(",")]

    # Load data
    if args.mode == "whisper":
        samples, normalizer, throughput_base = load_whisper_data(args)
        metric_fn = lambda p, r: compute_wer(p, r, normalizer)
        metric_name, throughput_name = "WER (%)", "Speed (x)"
        throughput_fn = lambda t: throughput_base / t if t > 0 else 0
    else:
        if args.mode == "translation":
            samples, tokenizer, throughput_base = load_translation_data(args)
        else:
            samples, tokenizer, throughput_base = load_generation_data(args)
        metric_fn = compute_bleu
        metric_name, throughput_name = "BLEU", "Tokens/sec"
        throughput_fn = lambda t: throughput_base / t if t > 0 else 0

    print(f"Mode: {args.mode} | Model: {args.model_path} | Samples: {len(samples)}")
    print(f"Runs: {args.warmup_runs} warmup + {args.num_runs} timed | CTranslate2: {ctranslate2.__version__}")
    print("=" * 90)

    results = []

    for device in devices:
        print(f"\nDEVICE: {device.upper()}")
        compute_types = sorted(ctranslate2.get_supported_compute_types(device))
        print(f"Compute types: {compute_types}")

        for ct in compute_types:
            print(f"\n  {ct}:")

            # Load model
            try:
                if args.mode == "whisper":
                    from faster_whisper import WhisperModel
                    model = WhisperModel(args.model_path, device=device, compute_type=ct)
                    run_fn = lambda: run_whisper(model, samples)
                elif args.mode == "translation":
                    model = ctranslate2.Translator(args.model_path, device=device, compute_type=ct)
                    run_fn = lambda: run_translation(model, samples, tokenizer)
                else:
                    model = ctranslate2.Generator(args.model_path, device=device, compute_type=ct)
                    run_fn = lambda: run_generation(model, samples, tokenizer, args)
            except RuntimeError as e:
                print(f"    Skipped: {e}")
                continue

            # Warmup
            for i in range(args.warmup_runs):
                t0 = time.time()
                run_fn()
                print(f"    Warmup {i+1}: {time.time()-t0:.2f}s")

            # Timed runs
            metrics, times = [], []
            for i in range(args.num_runs):
                t0 = time.time()
                preds, refs = run_fn()
                elapsed = time.time() - t0
                metric = metric_fn(preds, refs)
                metrics.append(metric)
                times.append(elapsed)
                print(f"    Run {i+1}: {metric_name}={metric:.2f} Time={elapsed:.2f}s")

            del model

            throughputs = [throughput_fn(t) for t in times]
            results.append({
                "device": device, "compute_type": ct,
                "metric": (np.mean(metrics), np.std(metrics)),
                "time": (np.mean(times), np.std(times)),
                "throughput": (np.mean(throughputs), np.std(throughputs)),
            })

            print(f"    → {metric_name}={np.mean(metrics):.2f}±{cv_percent(*results[-1]['metric']):.1f}% "
                  f"{throughput_name}={np.mean(throughputs):.2f}±{cv_percent(*results[-1]['throughput']):.1f}%")

    # Summary
    print("\n" + "=" * 90)
    print(f"{'Device':<8} {'Compute':<12} {metric_name:<20} {'Time (s)':<20} {throughput_name:<20}")
    print("-" * 90)
    for r in results:
        m, t, th = r["metric"], r["time"], r["throughput"]
        print(f"{r['device']:<8} {r['compute_type']:<12} "
              f"{m[0]:.2f}±{cv_percent(*m):.1f}%{'':<10} "
              f"{t[0]:.2f}±{cv_percent(*t):.1f}%{'':<10} "
              f"{th[0]:.2f}±{cv_percent(*th):.1f}%")


if __name__ == "__main__":
    main()
