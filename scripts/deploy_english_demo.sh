#!/usr/bin/env bash
# Deploy the public English demo (space_en_demo/): the app and the package. The weights are not copied: the app
# downloads them from the model repo (Vyvo/drifting-tts-en, scripts/prepare_english_release.py).
#   scripts/deploy_english_demo.sh <space id>
# The Space is created if it does not exist yet (public, Gradio); its hardware is set separately.
set -euo pipefail
cd "$(dirname "$0")/.."
SPACE=$1
STAGE=$(mktemp -d)
cp space_en_demo/app.py space_en_demo/requirements.txt space_en_demo/README.md "$STAGE/"
cp -r drifting_tts "$STAGE/"
find "$STAGE" -name __pycache__ -type d -prune -exec rm -rf {} +
hf repo create "$SPACE" --repo-type space --space-sdk gradio --public --exist-ok
hf upload "$SPACE" "$STAGE" . --repo-type space --commit-message "Deploy from github.com/kadirnar/drifting-tts@$(git rev-parse --short HEAD)"
rm -rf "$STAGE"
