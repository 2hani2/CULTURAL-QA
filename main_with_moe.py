"""
main_with_moe.py
-----------------
Always runs the pipeline WITH MoE re-scoring, regardless of whether a
checkpoint happens to auto-load correctly. Same CLI surface as main.py
(--ask, --questions_file, --dataset, --enable_images, interactive mode) --
this file adds nothing except pinning force_moe_rescore=True.

If no trained checkpoint is actually loaded, this still forces MoE on,
using RANDOM untrained weights -- pipeline.py prints a warning when that
happens. Train first (train_moe.py) if you want this to be meaningful,
not just technically "on".

Usage: identical to main.py, e.g.
  python main_with_moe.py
  python main_with_moe.py --ask "what is rangoli"
  python main_with_moe.py --questions_file questions.txt
"""

from main import run_cli

if __name__ == "__main__":
    run_cli(force_moe_rescore=True)