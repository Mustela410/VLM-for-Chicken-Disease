import json
import torch
import os
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from tqdm import tqdm

BASE_DIR = "/storage/student6/GalLens_student6"
TEST_FILE = os.path.join(BASE_DIR, "Model_eval/test_final_reduced.jsonl")
GT_LABELS_FILE = os.path.join(BASE_DIR, "Model_eval/gt_labels.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "Model_eval/model_answers")
IMAGE_ROOT = BASE_DIR

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
CACHE_DIR = "/storage/student6/mustela_cache"
LORA_PATH = "/storage/student6/final_lora_a100"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_base_model():
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, min_pixels=256*28*28, max_pixels=1280*28*28)
    return model, processor

def find_image_path(rel_path):
    basename = os.path.basename(rel_path)
    
    candidates = [
        os.path.join(IMAGE_ROOT, "image_test", basename),
        os.path.join(IMAGE_ROOT, rel_path),
        os.path.join(IMAGE_ROOT, rel_path.replace("Test/", "")),
        os.path.join(IMAGE_ROOT, rel_path.replace("Test/image_test/", "image_test/")),
        os.path.join(IMAGE_ROOT, "Dataset_images", basename),
        os.path.join(IMAGE_ROOT, rel_path.replace("Dataset_images/", ""))
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def run_inference(model, processor, data, gt_labels, model_name):
    output_file = os.path.join(OUTPUT_DIR, f"{model_name}.jsonl")
    
    processed_paths = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f_in:
            for line in f_in:
                if line.strip():
                    processed_paths.add(json.loads(line)['img_path'])
        print(f"Resuming {model_name}: {len(processed_paths)} already processed")
    
    mode = 'a' if processed_paths else 'w'
    skipped_count = 0
    processed_count = 0
    
    with open(output_file, mode) as f_out:
        for item in tqdm(data, desc=f"Running {model_name}"):
            img_path = item['img_path']
            
            if img_path in processed_paths:
                continue
            
            question = item['question']
            ground_truth = item['answer']
            
            image_path = find_image_path(img_path)
            if not image_path:
                skipped_count += 1
                continue
            
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": question}
                ]}
            ]
            
            text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            try:
                image_obj = Image.open(image_path).convert("RGB")
                inputs = processor(text=[text_prompt], images=[image_obj], padding=True, return_tensors="pt").to(DEVICE)
                
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
                
                generated_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
                prediction = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()
            except Exception as e:
                prediction = f"ERROR: {str(e)}"
            
            result = {
                "img_path": img_path,
                "question": question,
                "ground_truth": ground_truth,
                "model_answer": prediction,
                "gt_label": gt_labels.get(img_path, "healthy (feces)"),
                "model_version": model_name
            }
            f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
            processed_count += 1
    
    if skipped_count > 0:
        print(f"Skipped {skipped_count} samples (images not found)")
    print(f"Processed {processed_count} samples for {model_name}")

def main():
    with open(TEST_FILE, 'r') as f:
        test_data = [json.loads(line) for line in f if line.strip()]
    
    with open(GT_LABELS_FILE, 'r') as f:
        gt_labels = json.load(f)
    
    base_model, processor = load_base_model()
    
    print("Running base model...")
    run_inference(base_model, processor, test_data, gt_labels, "base_model")
    
    print("Running finetuned model...")
    if os.path.exists(LORA_PATH):
        finetuned_model = PeftModel.from_pretrained(base_model, LORA_PATH)
        finetuned_model.eval()
        run_inference(finetuned_model, processor, test_data, gt_labels, "finetuned_model")
        finetuned_model.unload()
    
    print("Done")

if __name__ == "__main__":
    main()
