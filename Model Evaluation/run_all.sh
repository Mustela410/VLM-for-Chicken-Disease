
set -e

echo "Step 1: Extract GT labels from img_path..."
python3 step1_extract_gt_labels.py

echo ""
echo "Step 2: Generate model answers (base + finetuned)..."
python3 step2_generate_model_answers.py

echo ""
echo "Step 3: Evaluate with BioBERT + Gemini..."
python3 step3_evaluate_with_gemini.py

echo ""
echo "Done"
