"""
main.py
-------
CLI for the CulturalQA-paper-style pipeline across 6 datasets.

Usage:
  python main.py --build
  python main.py --ask "question" [--dataset TAG]
  python main.py
  python main.py --questions_file questions.txt [--output results.csv] [--dataset TAG]
  python main.py --ablation_file questions.txt   # baseline vs MoE, side by side, one run
  python main.py --enable_images

force_moe_rescore: this file's own CLI leaves it at None (auto-detect --
MoE runs if a trained checkpoint loaded, same as always). The two
companion entry points, main_with_moe.py and main_without_moe.py, both
call run_cli() below with force_moe_rescore hardcoded to True / False --
same CLI surface (--ask, --questions_file, interactive, etc.), just with
MoE forced on or off regardless of checkpoint status. All three files
share this one implementation; nothing is duplicated between them.

GROQ_API_KEY: loaded automatically from a .env file in this folder if
present. Falls back to whatever's already in your shell environment if
no .env file exists.
"""

import os
import sys
import argparse

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from pipeline import build_index, ask, CulturalQAPipeline, run_questions_file, run_ablation_file

VALID_TAGS = ["SCALE", "DRISHTIKON", "DIWALI", "DOSA", "Durga QA", "Indian Cultural Dataset"]


def _check_groq_key():
    if not os.environ.get("GROQ_API_KEY"):
        print(
            "GROQ_API_KEY is not set.\n"
            "Either:\n"
            "  1. Create a .env file in this folder containing:\n"
            "       GROQ_API_KEY=your_key_here\n"
            "  2. Or run: export GROQ_API_KEY=your_key_here\n"
            "then try again."
        )
        sys.exit(1)


def chat_loop(dataset_filter: str = None, enable_images: bool = False, force_moe_rescore: bool = None):
    pipeline = CulturalQAPipeline(dataset_filter=dataset_filter, enable_images=enable_images,
                                   force_moe_rescore=force_moe_rescore)
    mode_tag = {True: " [MoE: FORCED ON]", False: " [MoE: FORCED OFF]", None: ""}[force_moe_rescore]
    print(f"CulturalQA (multimodal, MoE-fused, extractive) ready.{mode_tag} Type 'exit' to quit.")
    while True:
        q = input("\nAsk> ").strip()
        if q.lower() in ("exit", "quit"):
            break
        result = pipeline.ask(q)
        print("\n" + (result.get("answer") or result.get("message")))
        if result.get("path") == "exact_lookup":
            print(f"  (direct lookup: {result['dataset']} #{result['record_id']})")
        if "moe_rescore_used" in result:
            print(f"  (MoE re-scoring: {'ON' if result['moe_rescore_used'] else 'OFF'})")
        for r in result.get("retrieved_from", []):
            print(f"  from: {r['dataset']} ({r.get('source_file') or r.get('source_type')})")


def run_cli(force_moe_rescore: bool = None):
    """All the argument parsing + mode dispatch, as a reusable function.
    force_moe_rescore is baked in by the caller (None from this file's own
    __main__ block; True/False from main_with_moe.py/main_without_moe.py)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--ask", type=str, default=None)
    parser.add_argument("--questions_file", type=str, default=None,
                         help="Path to a text file with one question per line. "
                              "Runs all of them and writes a CSV of results.")
    parser.add_argument("--ablation_file", type=str, default=None,
                         help="Path to a text file with one question per line. Runs the "
                              "baseline-vs-MoE ablation on all of them in ONE pass, from the "
                              "exact same retrieved candidate pool per question -- always computes "
                              "both conditions regardless of force_moe_rescore, since that's the "
                              "whole point of this mode.")
    parser.add_argument("--output", type=str, default=None,
                         help="Output CSV path for --questions_file or --ablation_file mode.")
    parser.add_argument("--dataset", type=str, default=None, choices=VALID_TAGS)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--enable_images", action="store_true",
                         help="Turn on real CLIP encoding of DOSA/DRISHTIKON image URLs "
                              "(cached to disk after first fetch). Off by default.")
    args = parser.parse_args()

    _check_groq_key()

    if args.build:
        build_index(args.data_dir)
    elif args.ablation_file:
        run_ablation_file(args.ablation_file, output_path=args.output, dataset_filter=args.dataset,
                           enable_images=args.enable_images)
    elif args.questions_file:
        run_questions_file(args.questions_file, output_path=args.output, dataset_filter=args.dataset,
                            enable_images=args.enable_images, force_moe_rescore=force_moe_rescore)
    elif args.ask:
        ask(args.ask, dataset_filter=args.dataset, enable_images=args.enable_images,
            force_moe_rescore=force_moe_rescore)
    else:
        chat_loop(dataset_filter=args.dataset, enable_images=args.enable_images,
                  force_moe_rescore=force_moe_rescore)


if __name__ == "__main__":
    run_cli(force_moe_rescore=None)