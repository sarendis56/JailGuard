#!/usr/bin/env python3
"""
Unified JailGuard Evaluation Framework
Supports both text and image datasets with comprehensive metrics
"""

import argparse
import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import time
from datetime import datetime

# ML metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)

# Set GPU before importing torch (essential for CUDA)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

sys.path.append('./utils')
from utils import *
from mask_utils import *
from augmentations import *
import spacy
from PIL import Image
import shutil
from minigpt_utils import initialize_model, model_inference


class UnifiedJailGuardEvaluator:
    def __init__(self, args):
        self.args = args
        self.results = []
        self.metrics = {}
        
        # Load ground truth labels
        self.load_ground_truth()
        
        # Initialize model if needed
        if args.modality == 'image' or args.modality == 'text_with_minigpt':
            print("Initializing MiniGPT-4...")
            self.vis_processor, self.chat, self.model = initialize_model()
            print("MiniGPT-4 initialized successfully!")
        
        # Initialize spacy model for text similarity
        if args.modality in ['image', 'text_with_minigpt']:
            print("Loading spacy model...")
            self.nlp = spacy.load("en_core_web_md")
            print("Spacy model loaded!")
    
    def load_ground_truth(self):
        """Load ground truth labels for the dataset"""
        if self.args.modality == 'image':
            self.gt_labels = pickle.load(open('../dataset/image/dataset_key.pkl', 'rb'))
            self.dataset_size = 1000
            self.dataset_path = '../dataset/image/dataset'
        else:  # text datasets
            self.gt_labels = pickle.load(open('../dataset/text/dataset-key.pkl', 'rb'))
            self.dataset_size = 10000
            self.dataset_path = '../dataset/text'
            
        print(f"Loaded {len(self.gt_labels)} ground truth labels")
        
        # Convert labels to binary (1 for attack, 0 for benign)
        self.binary_labels = []
        for label_info in self.gt_labels:
            label = label_info[0].lower()
            if label == 'benign':
                self.binary_labels.append(0)
            else:
                self.binary_labels.append(1)  # All non-benign are attacks
                
        print(f"Attack samples: {sum(self.binary_labels)}, Benign samples: {len(self.binary_labels) - sum(self.binary_labels)}")
    
    def get_method(self, method_name):
        """Get augmentation method for image processing"""
        try:
            return img_aug_dict[method_name]
        except:
            print(f'Unknown method: {method_name}')
            sys.exit(1)
    
    def process_image_sample(self, serial_num):
        """Process a single image sample"""
        data_path = os.path.join(self.dataset_path, str(serial_num))
        
        # Check if image exists
        image_path = os.path.join(data_path, 'image.bmp')
        if not os.path.exists(image_path):
            image_path = os.path.join(data_path, 'image.jpg')
        
        if not os.path.exists(image_path):
            return None, f"Image not found for sample {serial_num}"
        
        question_path = os.path.join(data_path, 'question')
        if not os.path.exists(question_path):
            return None, f"Question not found for sample {serial_num}"
        
        # Create temporary directories
        variant_dir = f"/tmp/jailguard_variants_{serial_num}"
        response_dir = f"/tmp/jailguard_responses_{serial_num}"
        os.makedirs(variant_dir, exist_ok=True)
        os.makedirs(response_dir, exist_ok=True)
        
        try:
            # Generate image variants
            method = self.get_method(self.args.mutator)
            for i in range(self.args.number):
                pil_img = Image.open(image_path)
                new_image = method(img=pil_img)
                
                # Save variant
                ext = '.bmp' if '.bmp' in image_path else '.jpg'
                target_path = os.path.join(variant_dir, f'{i}-{self.args.mutator}{ext}')
                new_image.save(target_path)
            
            # Copy question file
            shutil.copy(question_path, os.path.join(variant_dir, 'question'))
            
            # Get model responses
            variant_list, name_list = load_mask_dir(variant_dir)
            with open(question_path, 'r') as f:
                question = ''.join(f.readlines())
            
            responses = []
            for j, img_prompt_path in enumerate(variant_list):
                prompts_eval = [question, img_prompt_path]
                result = model_inference(self.vis_processor, self.chat, self.model, prompts_eval)
                responses.append(result)
            
            # Calculate divergence and detection
            max_div, jailbreak_keywords = update_divergence(
                responses, serial_num, response_dir, 
                select_number=self.args.number, metric=self.nlp, top_string=100
            )
            
            # Detection decision
            is_attack_detected = detect_attack(max_div, jailbreak_keywords, self.args.threshold)
            
            return {
                'serial_num': serial_num,
                'max_divergence': max_div,
                'jailbreak_keywords': jailbreak_keywords,
                'is_attack_detected': is_attack_detected,
                'ground_truth': self.binary_labels[serial_num],
                'responses': responses
            }, None
            
        finally:
            # Cleanup
            if os.path.exists(variant_dir):
                shutil.rmtree(variant_dir)
            if os.path.exists(response_dir):
                shutil.rmtree(response_dir)
    
    def process_text_sample_with_minigpt(self, serial_num):
        """Process text sample using MiniGPT-4 (treating text as image description)"""
        # Load text dataset
        text_data = pickle.load(open('../dataset/text/dataset.pkl', 'rb'))
        
        if serial_num >= len(text_data):
            return None, f"Sample {serial_num} not found in text dataset"
        
        raw_text_data = text_data[serial_num]
        
        # Handle different data formats in the dataset
        if isinstance(raw_text_data, list):
            # Convert list of dicts (chat format) to string
            if all(isinstance(item, dict) for item in raw_text_data):
                # Chat message format: [{'role': 'system', 'content': '...'}, {'role': 'user', 'content': '...'}]
                text_prompt = str(raw_text_data)  # Convert to string representation
            else:
                # List of strings
                text_prompt = ''.join(raw_text_data)
        elif isinstance(raw_text_data, str):
            text_prompt = raw_text_data
        else:
            # Any other type, convert to string
            text_prompt = str(raw_text_data)
        
        # Create a simple text image for MiniGPT-4 (we'll just use the text directly)
        # For this experiment, we'll create a temporary image with the text
        from PIL import Image, ImageDraw, ImageFont
        
        # Create temporary directories
        variant_dir = f"/tmp/jailguard_text_variants_{serial_num}"
        response_dir = f"/tmp/jailguard_text_responses_{serial_num}"
        os.makedirs(variant_dir, exist_ok=True)
        os.makedirs(response_dir, exist_ok=True)
        
        try:
            # Apply text mutations for each variant
            responses = []
            for i in range(self.args.number):
                # Apply text mutation based on the mutator
                if self.args.mutator in text_aug_dict:
                    try:
                        # Apply text augmentation
                        mutated_text_list = text_aug_dict[self.args.mutator]([text_prompt])
                        
                        # DEBUG: Show mutation result (only if mutation actually changed something)
                        if not self.args.quiet and str(mutated_text_list) != str([text_prompt]):
                            mutated_preview = ''.join(mutated_text_list) if isinstance(mutated_text_list, list) else str(mutated_text_list)
                            if len(mutated_preview) > 150:
                                mutated_preview = mutated_preview[:150] + "..."
                            print(f"🔧 Mutation {i}: {mutated_preview}")
                        
                        # Robust handling of different return types from text augmentation functions
                        def flatten_to_string(obj):
                            """Recursively flatten any nested structure to a string"""
                            try:
                                if isinstance(obj, str):
                                    return obj
                                elif isinstance(obj, (list, tuple)):
                                    # Handle nested structures more safely
                                    result = []
                                    for item in obj:
                                        flattened = flatten_to_string(item)
                                        if isinstance(flattened, str):
                                            result.append(flattened)
                                    return ''.join(result)
                                else:
                                    return str(obj)
                            except:
                                # If anything fails, convert to string as last resort
                                return str(obj)
                        
                        mutated_prompt = flatten_to_string(mutated_text_list).strip()
                        
                        # Validate the result
                        if not mutated_prompt or not isinstance(mutated_prompt, str):
                            mutated_prompt = text_prompt
                            
                    except Exception as e:
                        print(f"⚠️  Text mutation error for sample {serial_num}, variant {i}: {e}")
                        mutated_prompt = text_prompt  # Use original on error
                else:
                    # No mutation for non-text mutators or fallback
                    mutated_prompt = text_prompt
                
                # Create a simple white image as placeholder for MiniGPT-4
                img = Image.new('RGB', (224, 224), color='white')
                temp_img_path = os.path.join(variant_dir, f'temp_{i}.jpg')
                img.save(temp_img_path)
                
                # Final safety check: ensure mutated_prompt is a string
                if not isinstance(mutated_prompt, str):
                    mutated_prompt = str(mutated_prompt)
                
                # Use the mutated text as the question
                prompts_eval = [mutated_prompt, temp_img_path]
                result = model_inference(self.vis_processor, self.chat, self.model, prompts_eval)
                responses.append(result)
                
                # Save the mutated text for debugging
                with open(os.path.join(variant_dir, f'mutated_text_{i}.txt'), 'w') as f:
                    f.write(mutated_prompt)
            
            # Calculate divergence and detection
            max_div, jailbreak_keywords = update_divergence(
                responses, serial_num, response_dir,
                select_number=self.args.number, metric=self.nlp, top_string=100
            )
            
            # Detection decision
            is_attack_detected = detect_attack(max_div, jailbreak_keywords, self.args.threshold)
            
            # DEBUG: Concise debugging output
            if not self.args.quiet:
                gt_label = "ATTACK" if self.binary_labels[serial_num] == 1 else "BENIGN"
                pred_label = "ATTACK" if is_attack_detected else "BENIGN"
                print(f"🔍 Sample {serial_num}: GT={gt_label} | Pred={pred_label} | Div={max_div:.4f} | Thresh={self.args.threshold}")
                
                # Show a few response previews for context
                if len(responses) > 1:
                    print(f"   📊 Response diversity:")
                    for i, resp in enumerate(responses[:2]):  # Show first 2 responses
                        print(f"      [{i}]: {resp[:80]}...")
                    if len(responses) > 2:
                        print(f"      ... and {len(responses)-2} more responses")
            
            return {
                'serial_num': serial_num,
                'max_divergence': max_div,
                'jailbreak_keywords': jailbreak_keywords,
                'is_attack_detected': is_attack_detected,
                'ground_truth': self.binary_labels[serial_num],
                'text_prompt': text_prompt,
                'responses': responses
            }, None
            
        finally:
            # Cleanup
            if os.path.exists(variant_dir):
                shutil.rmtree(variant_dir)
            if os.path.exists(response_dir):
                shutil.rmtree(response_dir)
    
    def run_evaluation(self):
        """Run evaluation on the specified range of samples"""
        print(f"Starting evaluation for {self.args.modality} modality")
        print(f"Mutator: {self.args.mutator}, Samples: {self.args.start_idx}-{self.args.end_idx}")
        print(f"Threshold: {self.args.threshold}, Variants per sample: {self.args.number}")
        
        failed_samples = []
        
        # Determine sample range
        start_idx = self.args.start_idx
        end_idx = min(self.args.end_idx, self.dataset_size - 1)
        
        progress_bar = tqdm(range(start_idx, end_idx + 1), desc="Processing samples")
        for i, serial_num in enumerate(progress_bar):
            try:
                start_time = time.time()
                
                if self.args.modality == 'image':
                    result, error = self.process_image_sample(serial_num)
                elif self.args.modality == 'text_with_minigpt':
                    result, error = self.process_text_sample_with_minigpt(serial_num)
                else:
                    print(f"Unsupported modality: {self.args.modality}")
                    continue
                
                process_time = time.time() - start_time
                
                if error:
                    if not self.args.quiet:
                        print(f"\n❌ Error processing sample {serial_num}: {error}")
                    failed_samples.append(serial_num)
                    continue
                
                if result:
                    self.results.append(result)
                    
                    # Detailed progress information
                    if not self.args.quiet:
                        attack_detected = "🔴 ATTACK" if result['is_attack_detected'] else "🟢 BENIGN"
                        gt_label = "ATTACK" if result['ground_truth'] == 1 else "BENIGN"
                        correct = "✅" if (result['is_attack_detected'] and result['ground_truth'] == 1) or \
                                        (not result['is_attack_detected'] and result['ground_truth'] == 0) else "❌"
                        
                        print(f"\n📊 Sample {serial_num} | {process_time:.1f}s | "
                              f"GT: {gt_label} | Pred: {attack_detected} | {correct}")
                        print(f"   Divergence: {result['max_divergence']:.4f} | "
                              f"Threshold: {self.args.threshold} | "
                              f"Keywords: {result['jailbreak_keywords']}")
                    
                    # Update progress bar with running accuracy
                    if len(self.results) > 0:
                        running_acc = sum(1 for r in self.results 
                                        if (r['is_attack_detected'] and r['ground_truth'] == 1) or 
                                           (not r['is_attack_detected'] and r['ground_truth'] == 0)) / len(self.results)
                        progress_bar.set_postfix({
                            'acc': f"{running_acc:.3f}",
                            'samples': len(self.results),
                            'fails': len(failed_samples),
                            'time': f"{process_time:.1f}s"
                        })
                
            except Exception as e:
                if not self.args.quiet:
                    print(f"\n💥 Exception processing sample {serial_num}: {str(e)}")
                failed_samples.append(serial_num)
                continue
        
        print(f"Processed {len(self.results)} samples successfully")
        if failed_samples:
            print(f"Failed samples: {failed_samples}")
        
        # THRESHOLD CALIBRATION ANALYSIS
        if len(self.results) > 5 and not self.args.quiet:
            self.analyze_threshold_performance()
    
    def analyze_threshold_performance(self):
        """Analyze divergence distribution and suggest optimal thresholds"""
        print("\n" + "="*60)
        print("📊 THRESHOLD CALIBRATION ANALYSIS")
        print("="*60)
        
        # Extract divergence values by ground truth
        attack_divs = [r['max_divergence'] for r in self.results if r['ground_truth'] == 1]
        benign_divs = [r['max_divergence'] for r in self.results if r['ground_truth'] == 0]
        
        print(f"📈 Divergence Statistics:")
        if attack_divs:
            print(f"   Attack samples  (n={len(attack_divs)}): "
                  f"mean={np.mean(attack_divs):.4f}, "
                  f"std={np.std(attack_divs):.4f}, "
                  f"range=[{min(attack_divs):.4f}, {max(attack_divs):.4f}]")
        
        if benign_divs:
            print(f"   Benign samples  (n={len(benign_divs)}): "
                  f"mean={np.mean(benign_divs):.4f}, "
                  f"std={np.std(benign_divs):.4f}, "
                  f"range=[{min(benign_divs):.4f}, {max(benign_divs):.4f}]")
        
        # Test different thresholds
        if len(attack_divs) > 0 and len(benign_divs) > 0:
            print(f"\n🎯 Threshold Performance Analysis:")
            all_divs = [r['max_divergence'] for r in self.results]
            test_thresholds = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, np.median(all_divs)]
            
            print(f"   Current threshold: {self.args.threshold:.4f}")
            print(f"   Threshold | Accuracy | TPR   | FPR   | F1    ")
            print(f"   ----------|----------|-------|-------|-------")
            
            for th in test_thresholds:
                tp = sum(1 for d in attack_divs if d > th)
                fn = len(attack_divs) - tp
                tn = sum(1 for d in benign_divs if d <= th) 
                fp = len(benign_divs) - tn
                
                if (tp + fn) > 0 and (tn + fp) > 0:
                    tpr = tp / (tp + fn)
                    fpr = fp / (tn + fp)
                    acc = (tp + tn) / (tp + tn + fp + fn)
                    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
                    
                    marker = " ⭐" if th == self.args.threshold else "   "
                    print(f"   {th:8.4f}{marker} | {acc:8.3f} | {tpr:5.3f} | {fpr:5.3f} | {f1:5.3f}")
        
        print("="*60)
    
    def calculate_metrics(self):
        """Calculate comprehensive evaluation metrics"""
        if not self.results:
            print("No results to calculate metrics!")
            return
        
        # Extract predictions and ground truth
        y_true = [r['ground_truth'] for r in self.results]
        y_pred_binary = [1 if r['is_attack_detected'] else 0 for r in self.results]
        y_scores = [r['max_divergence'] for r in self.results]  # Use divergence as confidence score
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred_binary)
        precision = precision_score(y_true, y_pred_binary, zero_division=0)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)
        
        # Confusion matrix - handle edge cases
        cm = confusion_matrix(y_true, y_pred_binary)
        if cm.size == 1:
            # Handle case where only one class is present
            unique_true = set(y_true)
            unique_pred = set(y_pred_binary)
            if len(unique_true) == 1 and len(unique_pred) == 1:
                true_label = list(unique_true)[0]
                pred_label = list(unique_pred)[0]
                if true_label == 1 and pred_label == 1:
                    tn, fp, fn, tp = 0, 0, 0, cm[0, 0]  # All true positives
                elif true_label == 0 and pred_label == 0:
                    tn, fp, fn, tp = cm[0, 0], 0, 0, 0  # All true negatives
                elif true_label == 1 and pred_label == 0:
                    tn, fp, fn, tp = 0, 0, cm[0, 0], 0  # All false negatives
                else:  # true_label == 0 and pred_label == 1
                    tn, fp, fn, tp = 0, cm[0, 0], 0, 0  # All false positives
            else:
                tn, fp, fn, tp = 0, 0, 0, 0
        else:
            tn, fp, fn, tp = cm.ravel()
        
        # Rates
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (Sensitivity/Recall)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate (Specificity)
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
        
        # ROC AUC and PR AUC (using divergence scores)
        try:
            roc_auc = roc_auc_score(y_true, y_scores)
        except:
            roc_auc = 0.5
        
        try:
            pr_auc = average_precision_score(y_true, y_scores)
        except:
            pr_auc = 0.0
        
        self.metrics = {
            'dataset_info': {
                'modality': self.args.modality,
                'mutator': self.args.mutator,
                'threshold': self.args.threshold,
                'samples_processed': len(self.results),
                'total_samples': self.args.end_idx - self.args.start_idx + 1
            },
            'confusion_matrix': {
                'true_negative': int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn),
                'true_positive': int(tp)
            },
            'primary_metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'roc_auc': float(roc_auc),
                'pr_auc': float(pr_auc)
            },
            'detailed_rates': {
                'true_positive_rate': float(tpr),
                'false_positive_rate': float(fpr),
                'true_negative_rate': float(tnr),
                'false_negative_rate': float(fnr)
            },
            'class_distribution': {
                'ground_truth_attacks': int(sum(y_true)),
                'ground_truth_benign': int(len(y_true) - sum(y_true)),
                'predicted_attacks': int(sum(y_pred_binary)),
                'predicted_benign': int(len(y_pred_binary) - sum(y_pred_binary))
            }
        }
    
    def save_results(self):
        """Save detailed results and metrics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"results_{self.args.modality}_{self.args.mutator}_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        
        # Save detailed results
        with open(os.path.join(results_dir, 'detailed_results.json'), 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save metrics
        with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Save summary CSV
        df_results = pd.DataFrame(self.results)
        df_results.to_csv(os.path.join(results_dir, 'results.csv'), index=False)
        
        print(f"Results saved to {results_dir}/")
        return results_dir
    
    def print_summary(self):
        """Print evaluation summary"""
        if not self.metrics:
            print("No metrics calculated!")
            return
        
        print("\n" + "="*80)
        print("JAILGUARD EVALUATION SUMMARY")
        print("="*80)
        
        # Dataset info
        info = self.metrics['dataset_info']
        print(f"Modality: {info['modality']}")
        print(f"Mutator: {info['mutator']}")
        print(f"Threshold: {info['threshold']}")
        print(f"Samples Processed: {info['samples_processed']}/{info['total_samples']}")
        
        # Primary metrics
        metrics = self.metrics['primary_metrics']
        print(f"\nPRIMARY METRICS:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1_score']:.4f}")
        print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
        print(f"  PR AUC:    {metrics['pr_auc']:.4f}")
        
        # Confusion matrix
        cm = self.metrics['confusion_matrix']
        print(f"\nCONFUSION MATRIX:")
        print(f"              Predicted")
        print(f"              Benign  Attack")
        print(f"Actual Benign   {cm['true_negative']:4d}    {cm['false_positive']:4d}")
        print(f"       Attack   {cm['false_negative']:4d}    {cm['true_positive']:4d}")
        
        # Rates
        rates = self.metrics['detailed_rates']
        print(f"\nDETAILED RATES:")
        print(f"  True Positive Rate (TPR):  {rates['true_positive_rate']:.4f}")
        print(f"  False Positive Rate (FPR): {rates['false_positive_rate']:.4f}")
        print(f"  True Negative Rate (TNR):  {rates['true_negative_rate']:.4f}")
        print(f"  False Negative Rate (FNR): {rates['false_negative_rate']:.4f}")
        
        # Class distribution
        dist = self.metrics['class_distribution']
        print(f"\nCLASS DISTRIBUTION:")
        print(f"  Ground Truth - Attacks: {dist['ground_truth_attacks']}, Benign: {dist['ground_truth_benign']}")
        print(f"  Predictions  - Attacks: {dist['predicted_attacks']}, Benign: {dist['predicted_benign']}")
        
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Unified JailGuard Evaluation Framework')
    
    # Core parameters
    parser.add_argument('--modality', choices=['image', 'text_with_minigpt'], default='image',
                        help='Evaluation modality (image uses visual data, text_with_minigpt tests MiniGPT-4 on text)')
    parser.add_argument('--mutator', default='PL', type=str,
                        help='Mutator method (HF,VF,RR,CR,RM,RS,GR,BL,CJ,RP,PL)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Detection threshold (auto-set: 0.025 for images, 0.02 for text, or specify manually)')
    parser.add_argument('--number', type=int, default=8,
                        help='Number of variants to generate per sample')
    
    # Sample range
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Start index for evaluation')
    parser.add_argument('--end_idx', type=int, default=99,
                        help='End index for evaluation (inclusive)')
    
    # Output options
    parser.add_argument('--save_results', action='store_true', default=True,
                        help='Save detailed results to files')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress detailed output')
    
    args = parser.parse_args()
    
    # Adjust threshold based on modality if not explicitly set
    # Auto-set threshold based on modality if not specified
    if args.threshold is None:
        if args.modality == 'text_with_minigpt':
            args.threshold = 0.1  # Higher threshold for MiniGPT-4 text experiments
        else:  # image modality
            args.threshold = 0.025
        print(f"Auto-set threshold to {args.threshold} for {args.modality} modality")
    
    # Initialize evaluator
    evaluator = UnifiedJailGuardEvaluator(args)
    
    # Run evaluation
    evaluator.run_evaluation()
    
    # Calculate metrics
    evaluator.calculate_metrics()
    
    # Print summary
    if not args.quiet:
        evaluator.print_summary()
    
    # Save results
    if args.save_results:
        results_dir = evaluator.save_results()
        print(f"\nDetailed results saved to: {results_dir}")


if __name__ == '__main__':
    main()
