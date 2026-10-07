"""
main_without_moe.py
--------------------
Always runs the pipeline WITHOUT MoE re-scoring -- relevance-filtering
only (retrieval + query expansion + cross-encoder rerank), the baseline
condition. Same CLI surface as main.py -- this file adds nothing except
pinning force_moe_rescore=False.

Path A (exact lookup) is completely unaffected either way -- MoE was
never in that loop regardless of this setting.

Usage: identical to main.py, e.g.
  python main_without_moe.py
  python main_without_moe.py --ask "what is rangoli"
  python main_without_moe.py --questions_file questions.txt
"""

from main import run_cli

if __name__ == "__main__":
    run_cli(force_moe_rescore=False)