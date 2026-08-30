#!/usr/bin/env bash
# One-shot publish. Pre-flight checks first, because pushing a repo that does
# not run is worse than pushing nothing.
set -euo pipefail

REPO_URL="${1:-}"
if [[ -z "$REPO_URL" ]]; then
  echo "usage: ./push_to_github.sh https://github.com/<you>/loom-twin.git"
  exit 1
fi

echo "==> pre-flight"
python -c "import numpy, pandas, scipy, sklearn, networkx" \
  || { echo "FAIL: run pip install -r requirements.txt"; exit 1; }

python run_demo.py --fast >/dev/null 2>&1 \
  || { echo "FAIL: run_demo.py errored"; exit 1; }
echo "    demo runs"

python eval/run_eval.py --seeds 2 >/dev/null 2>&1 \
  || { echo "FAIL: harness reported a metric regression"; exit 1; }
echo "    harness passes"

for f in README.md requirements.txt LICENSE .gitignore run_demo.py; do
  [[ -f "$f" ]] || { echo "FAIL: missing $f"; exit 1; }
done
echo "    required files present"

echo "==> publishing"
git init -q 2>/dev/null || true
git add -A
git -c user.email="${GIT_EMAIL:-team@example.com}" \
    -c user.name="${GIT_NAME:-Loom Team}" \
    commit -q -m "Loom: a digital twin for assembly lines with uneven sensor coverage" \
  || echo "    nothing new to commit"
git branch -M main
git remote remove origin 2>/dev/null || true
git remote add origin "$REPO_URL"
git push -u origin main

echo
echo "==> done. Now:"
echo "    1. Settings > General > check 'Public'"
echo "    2. Add the demo video link to the top of README.md"
echo "    3. Confirm the Actions tab shows a green run"
