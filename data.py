# Get necessary datasets from HuggingFace

from datasets import load_dataset, DownloadConfig # v2.22.0 Note: using v3 produces fsspec.exceptions.FSTimeoutError.

def get_dataset(save_dir, datasets_name):

    dlConfig = DownloadConfig(
        cache_dir = save_dir,
        resume_download=True,
        max_retries=100
    )

    # Load the dataset
    dataset = load_dataset(
        path = datasets_name,
        download_config=dlConfig,
        trust_remote_code=True,
    )

    return dataset

def main():

    ds1 = get_dataset("./charades", "HuggingFaceM4/charades")

    print("Done downloading")

    print(ds1['train'])

if __name__ == "__main__":
    main()