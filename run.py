"""One-command refresh for the project, so figures and results never silently
drift from their inputs again (the cooling-break numbers and the live trackers
both got stale this way: a stage was re-run without the stages that depend on it).

    python run.py thesis    # re-sync the cooling-break thesis (offline, no network)
    python run.py predict   # force-refresh the FIFA calendar, then ratings/sim/backtest
    python run.py live      # refresh the live break tracker, eval bars and match reports
    python run.py all       # thesis + predict + live, in dependency order

Each stage is a subprocess; the run stops at the first failure. `board.py` (the
per-match CV dossier) is intentionally left out -- it needs per-match clip frames
and the heavy vision deps, so run it by hand once the clips are in place.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

# Each phase is an ordered list of (script, *args). Order is the dependency order.
PHASES = {
    # offline: the grid feeds the analysis, the analysis feeds the figures
    "thesis": [
        ("src/xt_model.py",),
        ("src/analyze.py",),
        ("src/make_figures.py",),
    ],
    # fixtures.py force-refreshes the calendar first so the simulator and the
    # backtest see every finished match, not a stale cache
    "predict": [
        ("src/fixtures.py",),
        ("src/ratings.py",),
        ("src/montecarlo.py",),
        ("src/backtest.py",),
    ],
    "live": [
        ("src/wc2026.py",),
        ("src/live_eval.py",),
        ("src/replay.py",),
    ],
}
ORDER = ["thesis", "predict", "live"]


def run_step(step):
    print("\n=== {} ===".format(" ".join(step)), flush=True)
    subprocess.run([sys.executable, *step], cwd=ROOT, check=True)


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg == "all":
        phases = ORDER
    elif arg in PHASES:
        phases = [arg]
    else:
        print("usage: python run.py [thesis|predict|live|all]")
        sys.exit(2)
    for ph in phases:
        print("\n########## {} ##########".format(ph.upper()), flush=True)
        for step in PHASES[ph]:
            run_step(step)
    print("\nall stages completed", flush=True)


if __name__ == "__main__":
    main()
