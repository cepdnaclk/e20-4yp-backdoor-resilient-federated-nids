import kagglehub

path = kagglehub.dataset_download(
    "yasserhessein/cic-unsw-nb15-augmented-dataset"
)

print("Path to dataset files:", path)
