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

## Troubleshooting

If you can't get it to work, Linus can add you to the GCP project.

If the data is not showing up, try:

```bash
# Check DVC status
uv run dvc status

# Force re-pull
uv run dvc pull --force
```

## Updating data

```bash
# After adding/modifying images
uv run dvc add data/drone data/bird
git add data/drone.dvc data/bird.dvc
git commit -m "Update dataset"
uv run dvc push
git push
