# Frozen controlled pilot

Read PROTOCOL.md before running. These assets use preloaded atomic teaching,
one or two query generations, and validated structured teacher branches. They
do not reuse the older narrative-task call budget or establish real strategy
effects merely by passing fake tests.

Freeze after any intentional asset/code change, then verify and test:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.strategy_stability.controlled_v1.freeze --write
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest experiments/strategy_stability/tests experiments/strategy_stability/controlled_v1/tests -q -p no:cacheprovider
```

Run evidence must be outside the repository and the target directory must be
new. Fake-seek exercises 44 synthetic generations; fake-stop exercises 40.

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.strategy_stability.controlled_v1.runner --mode fake-seek --output /private/tmp/new-controlled-fake-run
```

An explicit `--mode real` uses the existing configured model credentials
without displaying them. It permits at most 44 calls and no retries. The
requested model must match frozen conditions.json. Use only after offline
acceptance and authorized real-call scope. A new experimental SQLite is
created in tmp and disposed after the run; results retain full audit evidence.

Read summary.json together with candidate-records.json and results.json.
Rendered compliance alone is not raw model compliance. The manifest includes
the actual verifier, language guard, isolated integration code and LLM client.
