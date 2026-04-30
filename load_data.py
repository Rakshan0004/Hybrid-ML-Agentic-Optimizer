import json
import glob
import os
import sys
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

# Force UTF-8 encoding for Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    repo_id = "netsol/resume-score-details"
    print(f"Loading sample records from dataset '{repo_id}'...")
    
    try:
        # Get list of files in the repo
        all_files = list_repo_files(repo_id, repo_type="dataset")
        json_files = [f for f in all_files if f.endswith('.json')]
        
        print(f"Found {len(json_files)} JSON files in the dataset.")
        
        # Download and load the first 5 files
        records = []
        num_to_load = min(5, len(json_files))
        
        for i in range(num_to_load):
            filename = json_files[i]
            # print(f"Downloading {filename}...")
            file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                records.append(data)
        
        # Convert to DataFrame for pretty printing
        df = pd.DataFrame(records)
        
        print(f"\nSuccessfully loaded {len(records)} records.")
        
        # Displaying with better formatting
        pd.set_option('display.max_colwidth', 50)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        
        print("\n--- Summary of Records ---")
        # Print basic info
        for idx, row in df.iterrows():
            print(f"\nRecord {idx + 1}:")
            if 'details' in row and isinstance(row['details'], dict):
                details = row['details']
                print(f"  Name: {details.get('name', 'N/A')}")
                print(f"  Email: {details.get('email', 'N/A')}")
                
                # Print skills
                skills = details.get('skills', [])
                if skills:
                    print(f"  Skills: {', '.join(skills[:5])}{'...' if len(skills) > 5 else ''}")
            
            if 'output' in row and isinstance(row['output'], dict):
                output = row['output']
                if 'scores' in output and 'total_score' in output['scores']:
                    print(f"  Total Score: {output['scores']['total_score']}")
                elif 'scores' in output and isinstance(output['scores'], dict) and 'total_score' in output['scores']:
                     print(f"  Total Score: {output['scores']['total_score']}")
            
            # Print first few characters of justification if it exists
            if 'output' in row and isinstance(row['output'], dict) and 'justification' in row['output']:
                just = row['output']['justification']
                if isinstance(just, list) and len(just) > 0:
                    print(f"  Justification: {just[0][:100]}...")
            
            # Print a snippet of the raw resume
            if 'input' in row and isinstance(row['input'], dict) and 'resume' in row['input']:
                resume_text = row['input']['resume']
                # Clean up newlines for cleaner printing
                clean_resume = " ".join(resume_text.split())
                print(f"  Resume Snippet: {clean_resume[:150]}...")

        print("\n--- Full Data Preview (First 2 records) ---")
        print(df.head(2))

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
