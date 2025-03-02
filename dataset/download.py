from datasets import load_dataset

rtl_repo = load_dataset('ahmedallam/RTL-Repo')

output_dir = "~/Workspace/pingqing2024/dataset/rtl-repo/"

for split in rtl_repo.keys():
    rtl_repo[split].to_json(f"{output_dir}/{split}.json", orient="records", lines=True)
