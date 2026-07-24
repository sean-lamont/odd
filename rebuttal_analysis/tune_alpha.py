"""TPE tuning of ODD's alpha on a small problem subset.

Wraps a sweep harness (sweep_mbpp.py etc.) as a black box: each trial runs the
harness at one float alpha on --n-problems problems, reads Pass@16 from the
newest *_metrics.json, and reports it to Optuna (sqlite storage, resumable).

Example:
  python rebuttal_analysis/tune_alpha.py --script sweep_mbpp.py \
      --n-problems 50 --temperature 1.0 --trials 12 --model-config llada
"""
import argparse, glob, json, os, subprocess, sys, time

import optuna


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True, help="harness script, e.g. sweep_mbpp.py")
    ap.add_argument("--n-problems", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--alpha-min", type=float, default=1.0)
    ap.add_argument("--alpha-max", type=float, default=160.0)
    ap.add_argument("--model-config", default="llada")
    ap.add_argument("--gen-length", type=int, default=None)
    ap.add_argument("--storage", default="sqlite:///tune_alpha.db")
    ap.add_argument("--study-name", default=None)
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    bench = os.path.basename(args.script).replace("sweep_", "").replace("_plain", "").replace(".py", "")
    study_name = args.study_name or f"{bench}_{args.model_config}_T{args.temperature}"
    study = optuna.create_study(
        study_name=study_name, storage=args.storage, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=0), load_if_exists=True)

    def objective(trial):
        alpha = trial.suggest_float("alpha", args.alpha_min, args.alpha_max, log=True)
        t0 = time.time()
        cmd = [sys.executable, args.script,
               "--strategies", "odd", "--alphas", str(alpha),
               "--temperatures", str(args.temperature),
               "--n-problems", str(args.n_problems),
               "--model-config", args.model_config]
        if args.gen_length:
            cmd += ["--gen-length", str(args.gen_length)]
        env = dict(os.environ, WANDB_MODE="offline")
        subprocess.run(cmd, check=True, env=env)
        # newest metrics file created after this trial started
        cands = [f for f in glob.glob(os.path.join(args.results_dir, "*", "*_metrics.json"))
                 if os.path.getmtime(f) >= t0]
        if not cands:
            raise RuntimeError("no metrics json produced by trial run")
        with open(max(cands, key=os.path.getmtime)) as fh:
            m = json.load(fh)
        p16 = m.get("pass_at_16")
        print(f"[tune] alpha={alpha:.2f} -> pass@16={p16}")
        return p16

    study.optimize(objective, n_trials=args.trials)
    best = study.best_trial
    print(f"[tune] BEST alpha={best.params['alpha']:.2f} pass@16={best.value:.4f} ({len(study.trials)} trials)")
    with open(f"tune_result_{study_name}.json", "w") as fh:
        json.dump({"study": study_name, "best_alpha": best.params["alpha"],
                   "best_pass_at_16": best.value,
                   "trials": [{"alpha": t.params.get("alpha"), "pass_at_16": t.value}
                              for t in study.trials if t.value is not None]}, fh, indent=2)


if __name__ == "__main__":
    main()
