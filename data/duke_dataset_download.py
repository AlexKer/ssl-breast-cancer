import kagglehub

# Download latest version
path = kagglehub.dataset_download("madhava20217/duke-breast-cancer-mri-nifti-pre-and-post-1-only")

print("Path to dataset files:", path)