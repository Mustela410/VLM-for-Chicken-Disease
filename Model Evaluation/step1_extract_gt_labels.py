import json
import os
import google.generativeai as genai
import time
import re
from tqdm import tqdm

BASE_DIR = "/storage/student6/GalLens_student6"
TEST_FILE = os.path.join(BASE_DIR, "Model_eval/test_final_reduced.jsonl")
OUTPUT_FILE = os.path.join(BASE_DIR, "Model_eval/gt_labels.json")

API_KEYS = []

DISEASE_CLASSES = [
    "avian influenza (head)", "chronic respiratory(head)",
    "healthy (feces)", "salmonella(feces)",
    "bumble foot", "healthy foot", "foot scaly leg mite", "foot spur",
    "fowlpox (head)", "healthy head",
    "new castle (feces)", "new castle disease(head)"
]

BATCH_SIZE = 120
MODEL_NAME = "gemini-2.5-flash"

class APIKeyManager:
    def __init__(self, keys):
        self.keys = keys
        self.current_idx = 0
    
    def get_next_key(self):
        key = self.keys[self.current_idx]
        self.current_idx = (self.current_idx + 1) % len(self.keys)
        return key

key_manager = APIKeyManager(API_KEYS)

def extract_gt_labels_batch(batch_paths, retries=5):
    batch_input = [{"id": idx, "img_path": path} for idx, path in enumerate(batch_paths)]
    
    prompt = f"""You are an expert in chicken disease classification. Your task is to extract the ground truth disease label from image filenames. The filenames follow a naming pattern where disease types are abbreviated or shortened.

INPUT: {json.dumps(batch_input, ensure_ascii=False)}

VALID CLASSES: {json.dumps(DISEASE_CLASSES)}

FILENAME MAPPING PATTERNS (use these as guidelines, but apply your understanding for edge cases):
- test_avian_*.jpg -> "avian influenza (head)"
- test_crd_*.jpg -> "chronic respiratory(head)"
- test_fowlpox_*.jpg -> "fowlpox (head)"
- test_ncd_head_*.jpg -> "new castle disease(head)"
- test_ncd_*.jpg (without "head") -> "new castle (feces)"
- test_salmon_*.jpg -> "salmonella(feces)"
- test_bbf_*.jpg -> "bumble foot"
- test_slm_*.jpg -> "foot scaly leg mite"
- test_spur_*.jpg -> "foot spur"
- test_head_healthy_*.jpg -> "healthy head"
- test_leg_healthy_*.jpg OR test_foot_healthy_*.jpg -> "healthy foot"
- test_healthy_*.jpg (without head/leg/foot specifier) -> "healthy (feces)"

For any unusual or ambiguous cases, use your understanding of the pattern and context to map to the most appropriate class from the VALID CLASSES list.

OUTPUT JSON:
[
    {{"id": 0, "gt_label": "class_name"}},
    ...
]
"""
    
    for i in range(retries):
        try:
            current_key = key_manager.get_next_key()
            genai.configure(api_key=current_key)
            model = genai.GenerativeModel(MODEL_NAME, generation_config={"temperature": 0.0, "response_mime_type": "application/json"})
            
            response = model.generate_content(prompt)
            json_str = re.search(r'\[.*\]', response.text, re.DOTALL).group(0)
            results = json.loads(json_str)
            
            if len(results) == len(batch_paths):
                return results
        except Exception as e:
            time.sleep(3)
    
    return [{"id": i, "gt_label": "healthy (feces)"} for i in range(len(batch_paths))]

def main():
    with open(TEST_FILE, 'r') as f:
        data = [json.loads(line) for line in f if line.strip()]
    
    checkpoint_file = OUTPUT_FILE.replace(".json", "_checkpoint.json")
    
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            gt_labels = json.load(f)
        print(f"Resuming from checkpoint: {len(gt_labels)} labels already extracted")
    else:
        gt_labels = {}
    
    img_paths = [item['img_path'] for item in data]
    chunks = [img_paths[i:i + BATCH_SIZE] for i in range(0, len(img_paths), BATCH_SIZE)]
    
    all_results = []
    processed_paths = set(gt_labels.keys())
    
    for chunk_idx, chunk in enumerate(tqdm(chunks, desc="Extracting GT labels")):
        chunk_to_process = [p for p in chunk if p not in processed_paths]
        if not chunk_to_process:
            continue
        
        results = extract_gt_labels_batch(chunk_to_process)
        
        for i, path in enumerate(chunk_to_process):
            gt_labels[path] = results[i]['gt_label']
        
        if (chunk_idx + 1) % 3 == 0:
            with open(checkpoint_file, 'w') as f:
                json.dump(gt_labels, f, indent=2)
        
        time.sleep(2)
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(gt_labels, f, indent=2)
    
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    
    print(f"Saved GT labels: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
