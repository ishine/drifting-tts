#!/usr/bin/env bash
# Deploy the private English test Space (space_en/): the English ARDrift model with BigVGAN-v2 fine-tuned on its mels
# (docs/ENGLISH.md, #70) and Vocos v2. The app is space_test/english_app.py, staged as app.py. The test Space has no
# room for the vocoder: a Space stores at most 1 GB, every stored file version included.
#   scripts/deploy_english_space.sh <space id> <exported English ARDrift checkpoint> <english_voices.json> <bigvgan_ft.pt>
# The checkpoints go into the Space repo itself (the vocoder in fp16); the script refuses a Space that is not private.
set -euo pipefail
cd "$(dirname "$0")/.."
SPACE=$1 MODEL=$2 VOICES=$3 VOCODER=$4
python -c "import sys; from huggingface_hub import HfApi; sys.exit(0 if HfApi().space_info('$SPACE').private else
'$SPACE is not private: refusing to deploy')"
STAGE=$(mktemp -d)
cp space_en/requirements.txt space_en/README.md "$STAGE/"
cp space_test/english_app.py "$STAGE/app.py"
cp -r drifting_tts "$STAGE/"
find "$STAGE" -name __pycache__ -type d -prune -exec rm -rf {} +
cp "$MODEL" "$STAGE/ardrift_en.pt"
cp "$VOICES" "$STAGE/english_voices.json"
python -c "import sys, torch; from drifting_tts.publish import export_bigvgan
torch.save(export_bigvgan(torch.load(sys.argv[1], map_location='cpu', weights_only=False), half=True), sys.argv[2])" \
  "$VOCODER" "$STAGE/bigvgan_en.pt"
du -sh "$STAGE"
hf upload "$SPACE" "$STAGE" . --repo-type space --commit-message "Deploy from github.com/kadirnar/drifting-tts@$(git rev-parse --short HEAD)"
rm -rf "$STAGE"
