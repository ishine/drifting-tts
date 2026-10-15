#!/usr/bin/env bash
# Deploy the private Mimi test Space (space_mimi/): the English ARDrift model on Kyutai's Mimi latents (docs/MIMI.md)
# with Mimi's released weights beside it (their `mimi.` tensors, CC BY 4.0: no Hub token needed in the Space):
#   scripts/deploy_mimi_space.sh <space id> <exported Mimi ARDrift checkpoint> <english_voices.json>
# The checkpoint goes into the Space repo itself; the script refuses a Space that is not private.
set -euo pipefail
cd "$(dirname "$0")/.."
SPACE=$1 MODEL=$2 VOICES=$3
python -c "import sys; from huggingface_hub import HfApi; sys.exit(0 if HfApi().space_info('$SPACE').private else
'$SPACE is not private: refusing to deploy')"
STAGE=$(mktemp -d)
cp space_mimi/app.py space_mimi/requirements.txt space_mimi/README.md "$STAGE/"
cp -r drifting_tts "$STAGE/"
find "$STAGE" -name __pycache__ -type d -prune -exec rm -rf {} +
cp "$MODEL" "$STAGE/ardrift_en_mimi.pt"
cp "$VOICES" "$STAGE/english_voices.json"
python -c "from drifting_tts.latents.mimi import save_weights; save_weights('$STAGE/mimi.safetensors')"
hf upload "$SPACE" "$STAGE" . --repo-type space --commit-message "Deploy from github.com/kadirnar/drifting-tts@$(git rev-parse --short HEAD)"
rm -rf "$STAGE"
