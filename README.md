# Drone Detector MLOps

A drone detection framework with extra MLOps sauce

## Cool features

- [x] Automatically delete branches after PR is merged
- [x] `make pr` uses Claude Code with the gh-pull-requests skill to create a PR
- [x] `make test` runs tests with coverage
- [x] Use the `uv run train` command to run the training script
- [x] Set up pre-commit hooks for linting and formatting
- [x] Github Actions check to make sure PR branch is up to date with main before running other workflows to minimize runner usage
- [x] Load environment variables from Github Secrets

## Cloud Infrastructure

**Region:** All infrastructure is in `europe-north2`

### Storage Context Manager

We use a storage abstraction layer ([`storage.py`](src/drone_detector_mlops/utils/storage.py)) that handles local vs cloud paths transparently. Set `MODE=local` or `MODE=cloud` in [`settings`](src/drone_detector_mlops/utils/settings.py) - the same code works in both environments.

### GCS Buckets

- **`gs://drone-detection-mlops-data/`** - Training data
  - `/structured/` - Direct training access (drone/, bird/, splits/)
  - `/dvc-store/` - DVC versioning history (not used during training)
- **`gs://drone-detection-mlops-models/`** - Trained models
  - `/checkpoints/` - Production model storage

#### Update structured data - for example if we get new data

```bash
gsutil -m rsync -r -d data/drone gs://drone-detection-mlops-data/structured/drone

gsutil -m rsync -r -d data/bird gs://drone-detection-mlops-data/structured/bird

gsutil -m rsync -r -d data/splits gs://drone-detection-mlops-data/structured/splits
```

### Artifact Registry

- **`europe-north2-docker.pkg.dev/drone-detection-mlops/ml-containers`**
  - Stores Docker images for cloud training
  - Current image: `train:latest` (training container)
