import json
import os
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm
import re

def clean_text(text):
    if not text:
        return ""
    # Remove common problematic characters like bullet points
    text = text.replace('\uf0b7', ' ')
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    repo_id = "netsol/resume-score-details"
    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Fetching file list from {repo_id}...")
    all_files = list_repo_files(repo_id, repo_type="dataset")
    json_files = [f for f in all_files if f.endswith('.json')]
    print(f"Found {len(json_files)} JSON files.")
    
    records = []
    
    print("Downloading and processing files...")
    for filename in tqdm(json_files):
        try:
            file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Extracting relevant fields for training
                resume = data.get('input', {}).get('resume', "")
                jd = data.get('input', {}).get('job_description', "")
                
                # The score is in output['scores']['aggregated_scores']
                scores_obj = data.get('output', {}).get('scores', {})
                agg_scores = scores_obj.get('aggregated_scores', {})
                
                # Calculate average of macro and micro scores
                m_score = agg_scores.get('macro_scores')
                mi_score = agg_scores.get('micro_scores')
                
                if m_score is not None and mi_score is not None:
                    total_score = (float(m_score) + float(mi_score)) / 2.0
                    
                    if resume and jd:
                        records.append({
                            'resume_text': clean_text(resume),
                            'job_description': clean_text(jd),
                            'total_score': total_score / 10.0, # Normalize 1-10 to 0-1
                            'original_filename': filename
                        })
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    df = pd.DataFrame(records)
    print(f"\nProcessed {len(df)} valid records.")
    
    # Save the processed data
    output_path = "data/train_raw.parquet"
    df.to_parquet(output_path, index=False)
    print(f"Dataset saved to {output_path}")

if __name__ == "__main__":
    main()
