# Dataset

Drone vs Bird images from [Kaggle](https://www.kaggle.com/datasets/muhammadsaoodsarwar/drone-vs-bird).

The actual image files are not in git (too large). They're stored in Google Cloud Storage and managed by DVC.

- **GCP Bucket**: `gs://drone-detection-mlops-data/dvc-store/`
- **DVC tracking files**: `bird.dvc` and `drone.dvc` (these are in git)

## Get the data

```bash
# First time: authenticate with GCP
gcloud auth application-default login

# Pull the data
uv run dvc pull
```

Images will appear in `data/drone/` and `data/bird/`.

## Updating data

```bash
# After adding/modifying images
uv run dvc add data/drone data/bird
git add data/drone.dvc data/bird.dvc
git commit -m "Update dataset"
uv run dvc push
git push
```

## Overview

The dataset has 4,104 images across two classes: drone (60.9%) and bird (39.1%).

- **Format**: All JPEG
- **Image sizes**: Drone images average ~1977x1189px, bird images average ~331x243px
- **Splits**: Pre-defined stratified train/val/test (70/15/15) in `data/splits/` for reproducibility

The dataset uses stratified splitting to maintain class balance across all splits. Images require resizing when loading (typically 224x224) - this is handled via transforms in the dataloader.
