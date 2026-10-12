#!/usr/bin/env bash
# Deploy the private test Space (space_test/): one tab per model under test, with their files:
#   scripts/deploy_test_space.sh <space id> <ARDrift checkpoint> <ARDrift voices checkpoint> <MelBWE checkpoint> <voices.json>
# The ARDrift tab is the public demo's app (space/app.py, staged as demo_app.py) with the ARDrift checkpoint beside it.
# The checkpoints go into the Space repo itself; the script refuses a Space that is not private.
set -euo pipefail
cd "$(dirname "$0")/.."
SPACE=$1 ARDRIFT=$2 MODEL=$3 BWE=$4 VOICES=$5
python -c "import sys; from huggingface_hub import HfApi; sys.exit(0 if HfApi().space_info('$SPACE').private else
'$SPACE is not private: refusing to deploy')"
if grep -n '{{' space/app.py; then  # result placeholders of a release not filled in yet
  echo "fill in the {{...}} placeholders above before deploying" >&2; exit 1
fi
STAGE=$(mktemp -d)
cp space_test/app.py space_test/voices_app.py space_test/requirements.txt space_test/README.md "$STAGE/"
cp space/app.py "$STAGE/demo_app.py"
cp -r drifting_tts "$STAGE/"
find "$STAGE" -name __pycache__ -type d -prune -exec rm -rf {} +
cp "$ARDRIFT" "$STAGE/ardrift.pt"
cp "$MODEL" "$STAGE/ardrift_voices.pt"
cp "$BWE" "$STAGE/bwe.pt"
cp "$VOICES" "$STAGE/voices.json"
hf upload "$SPACE" "$STAGE" . --repo-type space --commit-message "Deploy from github.com/kadirnar/drifting-tts@$(git rev-parse --short HEAD)"
rm -rf "$STAGE"
