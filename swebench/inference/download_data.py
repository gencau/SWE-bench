from huggingface_hub import snapshot_download
from datasets import load_dataset
import dotenv as env

env.load_dotenv()

def get_dataset():
    # pull every file that belongs to the dataset into ./hf_datasets/SWE‑bench_Lite
    snapshot_download(
        repo_id="princeton-nlp/SWE-bench_Verified",
        repo_type="dataset",
        local_dir="./hf_datasets/SWE-bench_Verified",
        local_dir_use_symlinks=False,   # copy instead of symlink so you can move it later
    )

def get_repos():
    ds = load_dataset("./hf_datasets/SWE-bench_Verified", split="test")
    unique = {(r, c) for r, c in zip(ds["repo"], ds["base_commit"])}
    with open("repos_needed.txt", "w") as f:
        for repo, commit in sorted(unique):
            f.write(f"{repo},{commit}\n")
    print(f"Wrote {len(unique)} unique repo/commit pairs")


get_dataset()
get_repos()