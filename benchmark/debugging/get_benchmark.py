import unzip
from tqdm import tqdm
import os
import json
import gzip
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
version2 = os.path.join(current_dir, "version2.json.gz")
gz_data = unzip.read_json_gz(version2)

git_commit_sha = [
    "580275ce67d3c",
    "8fa92faeff082",
    "8b0f4b335c8bfe",
    "fb115220b0c85",
    "70ee773f4d609",
    "501f28bd72e600",
    "d914eb9becfd1",
    "5cdec953072ec",
    "6d74be2b6054d6",
    "03c95e8d9e6ac",
    "d1da33069134d",
    "4888ad4f759b8c",
    "03fbb03f78db0",
    "e0565a359cc68",
    "32f88a41d69dbb",
    "580275ce67d3c",
    "8fa92faeff082",
    "8b0f4b335c8bfe",
    "143df43e328c62",
    "fa08ac5cb64d0",
    "404ccb2a8f0c9",
    "cfcfde74dc778",
    "81ca47870f706",
    "320cf042cd26f9",
    "03fbb03f78db0",
    "e0565a359cc68",
    "32f88a41d69dbb",
    "9c922ae8c623c2",
    "e4d4ef71ceff1",
    "506a2e7170fcd",
    "4172d4d7c2a13",
    "a16e69421e16e",
    "2fa0bded39ea0f",
]
output_dir = Path(current_dir) / "benchmark_results"
output_dir.mkdir(exist_ok=True)

for json_data in tqdm(gz_data):
    commit_sha = unzip.filter_benchmark(json_data).lower()
    benchmark_outfile = f"{commit_sha}.json"
    file_path = output_dir / benchmark_outfile
    # if commit_sha[:13] in git_commit_sha or commit_sha[:14] in git_commit_sha:
    if any(sub in commit_sha for sub in git_commit_sha):
        with open(file_path, 'w') as f:
            json.dump(json_data, f, indent=4)

print(len(git_commit_sha))
