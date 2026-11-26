# MedHELM leaderboard export

Export the MedHELM React leaderboard as static files for nginx, S3, or any static host.

## Prerequisites

- Python 3.10+ and [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) (`gcloud`) for downloading public results
- This repository checked out and installed in a virtual environment
- Optional: Node.js and npm if you change `helm-frontend/`

## Setup

```bash
cd /path/to/medhelm
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
helm-server -h | grep export-path
```

Use the `helm-server` from this venv. A global or Homebrew install may not support `--export-path`.

## 1. Populate leaderboard data

Download the public MedHELM release tree from GCS:

```bash
export OUTPUT_PATH="./benchmark_output"
export GCS_BENCHMARK_OUTPUT_PATH="gs://crfm-helm-public/medhelm/benchmark_output"

mkdir -p "$OUTPUT_PATH"
gcloud storage rsync -r "$GCS_BENCHMARK_OUTPUT_PATH" "$OUTPUT_PATH"
```

Confirm release folders exist, for example:

- `benchmark_output/releases/v2.0.0/summary.json`
- `benchmark_output/releases/v1.0.0/summary.json`

To generate your own runs instead of downloading, use `medhelm-run` and `helm-summarize` against the same `OUTPUT_PATH`. See [docs/medhelm.md](docs/medhelm.md).

### Copy runs with only metrics and no instance and response data
```bash
rsync -av \
  --exclude='scenario_state.json' \
  --exclude='instances.json' \
  --exclude='display_requests.json' \
  --exclude='display_predictions.json' \
  --exclude='per_instance_stats.json' \
  . $OUTPUT_PATH
```

### Redact licensed scenario data but keep the instance view

Some scenarios are built from licensed data that must not be redistributed, and pass it in
via a separately downloaded `data_path` (currently **CPC-QA**, whose case records are licensed
from NEJM). A normal run publishes that data in full: `scenario_state.json` holds the case
text, the answer options and the prompt, the three `display_*.json` files are derived from it,
and `helm-server` serves everything under `OUTPUT_PATH` with no access check.

Excluding the files as above works, but drops the whole instance view and the per-instance
stats. To keep both while blanking the licensed text, redact the suite instead:

```bash
python scripts/medhelm/redact_instances.py --suite <suite> -o "$OUTPUT_PATH"
helm-summarize --suite <suite> -o "$OUTPUT_PATH"
python scripts/medhelm/redact_instances.py --suite <suite> -o "$OUTPUT_PATH" --check
```

The first command rewrites `scenario_state.json` in place, replacing the instance input, the
answer options and the request prompt with `[redacted]`. `helm-summarize` then regenerates the
display files from the redacted state. Scores cannot change: `stats.json` and
`per_instance_stats.json` are written during `medhelm-run`, before redaction, and stay
byte-identical.

`--check` is the gate. It is read-only and exits non-zero if any licensed string from the
source dataset still appears in any artifact. **Run it immediately before every step below** —
`--export-path`, `gcloud storage rsync` and starting `helm-server` on a reachable host all
publish whatever is on disk.

Notes:

- Redaction is destructive and in place. The unredacted artifacts only exist on the machine
  that ran the eval, so redact a copy if you still need the case text for debugging.
- It is idempotent, so it is safe to re-run on a suite that may already be redacted.
- It only touches scenarios listed in `RESTRICTED_SCENARIOS` in the script (`cpc_qa` today);
  override with `--scenarios`. Every other run in the suite passes through untouched.
- Model completions are kept so per-instance answers stay debuggable. For a scenario whose
  answers are single letters this is safe, but a reasoning model can quote the case back
  through its completion text; `--check` warns when a completion is suspiciously long.

## 2. Optional: rebuild the UI

Only needed after editing `helm-frontend/`:


```bash
cd /path/to/medhelm/helm-frontend
npm install --global yarn   # if you don’t already have Yarn
yarn install
yarn build --outDir '../src/helm/benchmark/static_build' --emptyOutDir
cd /path/to/medhelm
```
OR
```bash
cd helm-frontend
npm install
npm run build
cd ..

rm -rf src/helm/benchmark/static_build/assets
cp -r helm-frontend/dist/* src/helm/benchmark/static_build/
```

## 3. Run helm-summarize for multiple suites

```text
helm-summarize --release v5.0.0 --suites v4.0.0 meta_v2 meta_v1alpha -o benchmarks/ --schema-path schema_v5.yaml
```

## 4. Export static sites

Pick a site root and portal base URL (your public origin and path prefix). Export **one directory per release**. Each `--export-path` must not exist yet.

```bash
export SITE_ROOT="/tmp/medhelm_site"
export OUTPUT_PATH="./benchmark_output"
export PORTAL_BASE="https://your.domain/medhelm"

rm -rf "$SITE_ROOT"
mkdir -p "$SITE_ROOT/medhelm"

helm-server \
  --release v2.0.0 \
  -o "$OUTPUT_PATH" \
  --project medhelm \
  --export-path "$SITE_ROOT/medhelm/v2.0.0" \
  --helm-portal-base-url "$PORTAL_BASE" \
  --helm-project-metadata-url "$PORTAL_BASE/project_metadata.json"

helm-server \
  --release v1.0.0 \
  -o "$OUTPUT_PATH" \
  --project medhelm \
  --export-path "$SITE_ROOT/medhelm/v1.0.0" \
  --helm-portal-base-url "$PORTAL_BASE" \
  --helm-project-metadata-url "$PORTAL_BASE/project_metadata.json"
```

Repeat for other tags under `benchmark_output/releases/`.

## 5. Release menu metadata

The navbar release dropdown needs `project_metadata.json` at the URL you set above.

```bash
 cat > "$SITE_ROOT/project_metadata.json" <<'EOF'
[
  {
    "title": "MedHELM",
    "description": "MedHELM leaderboard",
    "id": "medhelm",
    "releases": ["v4.0.0", "v3.0.0", "v2.0.0", "v1.0.0"]
  }
]
EOF
```

`HELM_PROJECT_METADATA_URL` in each export’s `config.js` must match where this file is served. If the file lives under `/medhelm/`, do not point metadata at `/helm/`.

## 6. Preview locally

```bash
python -m http.server -d "$SITE_ROOT"
```

Open:

- `http://127.0.0.1:8000/medhelm/v2.0.0/`
- `http://127.0.0.1:8000/medhelm/project_metadata.json`

Hard-refresh if `config.js` is cached.

## 7. Deploy

Upload `$SITE_ROOT` to your host or map nginx locations to each version folder and to `medhelm/project_metadata.json`.

The static site cam be hosted with AWS s3 Static Syte and AWS CloudFront.

## 8. Update the homepage leaders

The landing page does not read the exported leaderboard. After each release, update the **Current leaders** panel in [docs/index.html](docs/index.html) so it matches the latest table.

Update only the leader rows and the footer:

- Model name, organization, description, and run link
- Mean win rate, including the version label (for example, `Mean win rate v6.0.0`)
- Footer count and date (for example, `10 of 14 models · Updated 23 Nov 2026`)

Leave the surrounding markup and styles unchanged.

