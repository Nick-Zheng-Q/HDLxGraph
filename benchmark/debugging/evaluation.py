from rouge import Rouge
import gzip
import json
import os
from tqdm import tqdm
from . import unzip

current_dir = os.path.dirname(os.path.abspath(__file__))
# Initialize the ROUGE evaluator
def rouge_evaluation(candidate,reference):
    rouge = Rouge()

    scores = rouge.get_scores(candidate, reference)
    return scores

def main_debugging(model, RAG, generate_func):
    dir = os.path.join(current_dir, "benchmark_results")
    json_data_list = unzip.read_json_files(dir)
    scores = []
    for json_data in tqdm(json_data_list):
        before, after, pull_body, pull_title = unzip.get_bug_info(json_data)
        bug = f"""
        pull_title:{pull_title}
        pull_body:{pull_body}
        """
        prompt = before + '\n' + bug
        prompt = RAG.retrieve(prompt)
        corrected_code = generate_func(model, "", prompt)
        score = rouge_evaluation(corrected_code, after)
        scores.append(score)

    score_output_filename = "scores.json.gz"
    with gzip.open(score_output_filename, 'wt') as f:
        json.dump(scores, f, indent=4)

    print(f"scores have been saved to {score_output_filename}")
    
if __name__ == "__main__":
    main_debugging()
    
