import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_from_disk
from pathlib import Path
from tqdm import tqdm
import json
import re
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter

# Configuration
MODEL_ID = "google/medgemma-27b-text-it"
DATASET_DIR = "hf_dataset"
PROMPT_FILE = "prompts/ip_op_prediction.txt"
RESULTS_FILE = "test_inference_results.json"
METRICS_FILE = "test_inference_metrics.txt"
MAX_RETRIES = 5
MAX_NEW_TOKENS = 10  # Short response expected: just "IP" or "OP"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_storage=torch.bfloat16,
)

def load_prompt_template():
    """Load the prompt template from file."""
    with open(PROMPT_FILE, 'r') as f:
        return f.read()

def prepare_prompt(template, clinical_notes):
    """Replace placeholder with actual clinical notes."""
    return template.replace('[collated_notes]', clinical_notes)

def extract_prediction(response_text):
    """
    Extract IP or OP from model response.
    Returns: 'IP', 'OP', or None if invalid.
    """
    # Remove whitespace and convert to uppercase
    clean_response = response_text.strip().upper()

    # Check for exact matches first
    if clean_response == "IP":
        return "IP"
    elif clean_response == "OP":
        return "OP"

    # Check if IP or OP appears in the response
    if "IP" in clean_response and "OP" not in clean_response:
        return "IP"
    elif "OP" in clean_response and "IP" not in clean_response:
        return "OP"

    # If both or neither found, return None (invalid)
    return None

def run_inference_with_retry(model, tokenizer, prompt, max_retries=MAX_RETRIES):
    """
    Run inference with retry mechanism.
    Returns: (prediction, num_attempts, success)
    """
    for attempt in range(1, max_retries + 1):
        try:
            # Prepare messages
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful medical assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            # Tokenize
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)

            input_len = inputs["input_ids"].shape[-1]

            # Generate
            with torch.inference_mode():
                generation = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    temperature=0,
                    top_p=None
                )
                generation = generation[0][input_len:]

            # Decode
            decoded = tokenizer.decode(generation, skip_special_tokens=True)

            # Extract prediction
            prediction = extract_prediction(decoded)

            if prediction is not None:
                return prediction, attempt, True

        except Exception as e:
            print(f"Error on attempt {attempt}: {str(e)}")
            continue

    # Failed after all retries
    return None, max_retries, False

def main():
    print("="*60)
    print("Test Set Inference Pipeline")
    print("="*60)

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_from_disk(DATASET_DIR)
    test_data = dataset['test']
    print(f"   Test set size: {len(test_data)}")

    # Load prompt template
    print("\nLoading prompt template...")
    prompt_template = load_prompt_template()
    print(f"   Loaded from: {PROMPT_FILE}")

    # Load model
    print("\nLoading model...")
    print(f"   Model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quantization_config
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print("   Model loaded successfully")

    # Run inference
    print("\nRunning inference on test set...")
    results = []
    failures = []

    for idx, sample in enumerate(tqdm(test_data, desc="Processing")):
        patient_id = sample['patient_id']
        true_class = sample['class']
        notes = sample['notes']

        # Prepare prompt
        prompt = prepare_prompt(prompt_template, notes)

        # Run inference with retry
        prediction, attempts, success = run_inference_with_retry(
            model, tokenizer, prompt, MAX_RETRIES
        )

        result = {
            'patient_id': patient_id,
            'true_class': true_class,
            'predicted_class': prediction,
            'attempts': attempts,
            'success': success,
            'num_visits': sample['num_visits']
        }

        results.append(result)

        if not success:
            failures.append({
                'patient_id': patient_id,
                'true_class': true_class,
                'attempts': attempts
            })

    # Save results
    print("\n5. Saving results...")
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   Results saved to: {RESULTS_FILE}")

    # Calculate metrics
    print("\n6. Calculating metrics...")
    successful_results = [r for r in results if r['success']]

    if len(successful_results) == 0:
        print("   ERROR: No successful predictions!")
        return

    y_true = [r['true_class'] for r in successful_results]
    y_pred = [r['predicted_class'] for r in successful_results]

    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=['IP', 'OP'])
    class_report = classification_report(y_true, y_pred, labels=['IP', 'OP'])

    # Calculate per-class accuracy
    class_counts = Counter(y_true)
    class_correct = Counter()
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            class_correct[true] += 1

    # Write metrics to file
    with open(METRICS_FILE, 'w') as f:
        f.write("="*60 + "\n")
        f.write("Test Set Inference Metrics\n")
        f.write("="*60 + "\n\n")

        f.write(f"Total test samples: {len(test_data)}\n")
        f.write(f"Successful predictions: {len(successful_results)}\n")
        f.write(f"Failed predictions: {len(failures)}\n")
        f.write(f"Success rate: {len(successful_results)/len(test_data)*100:.2f}%\n\n")

        f.write("Failed Samples:\n")
        for fail in failures:
            f.write(f"  Patient ID: {fail['patient_id']}, True Class: {fail['true_class']}, Attempts: {fail['attempts']}\n")
        f.write("\n")

        f.write(f"Overall Accuracy: {accuracy*100:.2f}%\n\n")

        f.write("Class-wise Performance:\n")
        for cls in ['IP', 'OP']:
            if cls in class_counts:
                cls_acc = class_correct[cls] / class_counts[cls] * 100 if class_counts[cls] > 0 else 0
                f.write(f"  {cls}: {class_correct[cls]}/{class_counts[cls]} = {cls_acc:.2f}%\n")
        f.write("\n")

        f.write("Confusion Matrix:\n")
        f.write("                Predicted\n")
        f.write("              IP    OP\n")
        f.write(f"True  IP    {conf_matrix[0][0]:4d}  {conf_matrix[0][1]:4d}\n")
        f.write(f"      OP    {conf_matrix[1][0]:4d}  {conf_matrix[1][1]:4d}\n\n")

        f.write("Classification Report:\n")
        f.write(class_report)
        f.write("\n")

        # Attempt statistics
        attempt_counts = Counter([r['attempts'] for r in results])
        f.write("Attempts Distribution:\n")
        for attempt in sorted(attempt_counts.keys()):
            f.write(f"  {attempt} attempt(s): {attempt_counts[attempt]} samples\n")

    print(f"   Metrics saved to: {METRICS_FILE}")

    # Print summary to console
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total samples: {len(test_data)}")
    print(f"Successful: {len(successful_results)} ({len(successful_results)/len(test_data)*100:.2f}%)")
    print(f"Failed: {len(failures)} ({len(failures)/len(test_data)*100:.2f}%)")
    print(f"\nOverall Accuracy: {accuracy*100:.2f}%")
    print("\nClass-wise Accuracy:")
    for cls in ['IP', 'OP']:
        if cls in class_counts:
            cls_acc = class_correct[cls] / class_counts[cls] * 100 if class_counts[cls] > 0 else 0
            print(f"  {cls}: {cls_acc:.2f}% ({class_correct[cls]}/{class_counts[cls]})")
    print("="*60)

if __name__ == "__main__":
    main()
