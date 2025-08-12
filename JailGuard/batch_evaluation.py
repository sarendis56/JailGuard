#!/usr/bin/env python3
"""
Batch Evaluation Script for JailGuard
Runs comprehensive evaluations across different mutators and modalities
"""

import argparse
import subprocess
import os
import json
import pandas as pd
from datetime import datetime
import time


def run_evaluation(modality, mutator, start_idx, end_idx, threshold, number, quiet=True):
    """Run a single evaluation experiment"""
    cmd = [
        'python', 'unified_evaluation.py',
        '--modality', modality,
        '--mutator', mutator,
        '--start_idx', str(start_idx),
        '--end_idx', str(end_idx),
        '--threshold', str(threshold),
        '--number', str(number),
        '--save_results'
    ]
    
    if quiet:
        cmd.append('--quiet')
    
    print(f"🚀 Running: {modality} | {mutator} | samples {start_idx}-{end_idx}")
    if not quiet:
        print(f"📝 Command: {' '.join(cmd)}")
    
    start_time = time.time()
    
    if not quiet:
        # Run with real-time output for better monitoring
        print("📊 Real-time progress:")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                 text=True, bufsize=1, universal_newlines=True)
        
        stdout_lines = []
        for line in process.stdout:
            print(f"   {line.rstrip()}")  # Prefix with spaces for clarity
            stdout_lines.append(line)
        
        process.wait()
        result_stdout = ''.join(stdout_lines)
        result_stderr = ""
        returncode = process.returncode
    else:
        result = subprocess.run(cmd, capture_output=True, text=True)
        result_stdout = result.stdout
        result_stderr = result.stderr
        returncode = result.returncode
    
    end_time = time.time()
    duration = end_time - start_time
    
    if returncode != 0:
        print(f"❌ ERROR in {modality}/{mutator} after {duration:.1f}s: {result_stderr if 'result_stderr' in locals() else 'Unknown error'}")
        return None
    else:
        print(f"✅ Completed in {duration:.1f}s ({duration/60:.1f} minutes)")
    
    # Use the collected stdout
    stdout_lines = result_stdout.strip().split('\n') if 'result_stdout' in locals() else []
    
    # Parse metrics from output
    try:
        # Find the results directory from stdout
        results_dir = None
        for line in stdout_lines:
            if 'Detailed results saved to:' in line:
                results_dir = line.split(': ')[-1]
                break
        
        if results_dir and os.path.exists(os.path.join(results_dir, 'metrics.json')):
            with open(os.path.join(results_dir, 'metrics.json'), 'r') as f:
                metrics = json.load(f)
            
            # Add timing info
            metrics['runtime_seconds'] = end_time - start_time
            metrics['results_directory'] = results_dir
            
            return metrics
    except Exception as e:
        print(f"Error parsing results for {modality}/{mutator}: {str(e)}")
    
    return None


def create_summary_report(all_results, output_dir):
    """Create comprehensive summary report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create summary DataFrame
    summary_data = []
    for result in all_results:
        if result is None:
            continue
            
        info = result['dataset_info']
        metrics = result['primary_metrics']
        rates = result['detailed_rates']
        
        summary_data.append({
            'modality': info['modality'],
            'mutator': info['mutator'],
            'threshold': info['threshold'],
            'samples': info['samples_processed'],
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'roc_auc': metrics['roc_auc'],
            'pr_auc': metrics['pr_auc'],
            'tpr': rates['true_positive_rate'],
            'fpr': rates['false_positive_rate'],
            'runtime_min': result.get('runtime_seconds', 0) / 60
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Save summary
    summary_file = os.path.join(output_dir, f'evaluation_summary_{timestamp}.csv')
    df_summary.to_csv(summary_file, index=False)
    
    # Save detailed results
    detailed_file = os.path.join(output_dir, f'detailed_results_{timestamp}.json')
    with open(detailed_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*100)
    print("BATCH EVALUATION SUMMARY")
    print("="*100)
    
    if len(df_summary) > 0:
        # Group by modality
        for modality in df_summary['modality'].unique():
            modal_df = df_summary[df_summary['modality'] == modality]
            print(f"\n{modality.upper()} RESULTS:")
            print("-" * 50)
            
            for _, row in modal_df.iterrows():
                print(f"{row['mutator']:3s} | Acc:{row['accuracy']:.3f} | "
                      f"F1:{row['f1_score']:.3f} | AUC:{row['roc_auc']:.3f} | "
                      f"TPR:{row['tpr']:.3f} | FPR:{row['fpr']:.3f} | "
                      f"Time:{row['runtime_min']:.1f}min")
            
            # Best performers (handle NaN values)
            best_acc = modal_df.loc[modal_df['accuracy'].idxmax()]
            best_f1 = modal_df.loc[modal_df['f1_score'].idxmax()]
            
            # Handle NaN ROC AUC values
            valid_auc = modal_df.dropna(subset=['roc_auc'])
            if len(valid_auc) > 0:
                best_auc = valid_auc.loc[valid_auc['roc_auc'].idxmax()]
                print(f"\nBest Accuracy:  {best_acc['mutator']} ({best_acc['accuracy']:.4f})")
                print(f"Best F1 Score:  {best_f1['mutator']} ({best_f1['f1_score']:.4f})")
                print(f"Best ROC AUC:   {best_auc['mutator']} ({best_auc['roc_auc']:.4f})")
            else:
                print(f"\nBest Accuracy:  {best_acc['mutator']} ({best_acc['accuracy']:.4f})")
                print(f"Best F1 Score:  {best_f1['mutator']} ({best_f1['f1_score']:.4f})")
                print(f"Best ROC AUC:   N/A (all values undefined)")
    
    print(f"\nDetailed results saved to:")
    print(f"  Summary: {summary_file}")
    print(f"  Details: {detailed_file}")
    print("="*100)
    
    return summary_file, detailed_file


def main():
    parser = argparse.ArgumentParser(description='Batch JailGuard Evaluation')
    
    # Experiment scope
    parser.add_argument('--modalities', nargs='+', 
                        choices=['image', 'text_with_minigpt'], 
                        default=['image'],
                        help='Modalities to evaluate')
    parser.add_argument('--mutators', nargs='+',
                        default=['PL', 'HF', 'VF', 'RR', 'CR'],
                        help='Mutators to test')
    
    # Sample range
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Start index for evaluation')
    parser.add_argument('--end_idx', type=int, default=99,
                        help='End index for evaluation')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size for processing (split large ranges)')
    
    # Parameters
    parser.add_argument('--number', type=int, default=8,
                        help='Number of variants per sample')
    parser.add_argument('--image_threshold', type=float, default=0.025,
                        help='Threshold for image detection')
    parser.add_argument('--text_threshold', type=float, default=0.02,
                        help='Threshold for text detection')
    
    # Output
    parser.add_argument('--output_dir', default='batch_results',
                        help='Output directory for results')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress individual experiment output')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate experiment configurations
    experiments = []
    for modality in args.modalities:
        threshold = args.image_threshold if modality == 'image' else args.text_threshold
        
        for mutator in args.mutators:
            # Split into batches if needed
            total_samples = args.end_idx - args.start_idx + 1
            if total_samples <= args.batch_size:
                experiments.append({
                    'modality': modality,
                    'mutator': mutator,
                    'start_idx': args.start_idx,
                    'end_idx': args.end_idx,
                    'threshold': threshold
                })
            else:
                # Split into batches
                for batch_start in range(args.start_idx, args.end_idx + 1, args.batch_size):
                    batch_end = min(batch_start + args.batch_size - 1, args.end_idx)
                    experiments.append({
                        'modality': modality,
                        'mutator': mutator,
                        'start_idx': batch_start,
                        'end_idx': batch_end,
                        'threshold': threshold
                    })
    
    print(f"Running {len(experiments)} experiments...")
    print(f"Modalities: {args.modalities}")
    print(f"Mutators: {args.mutators}")
    print(f"Sample range: {args.start_idx}-{args.end_idx}")
    print(f"Variants per sample: {args.number}")
    
    # Run experiments
    all_results = []
    total_time = time.time()
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] ", end="")
        
        result = run_evaluation(
            modality=exp['modality'],
            mutator=exp['mutator'], 
            start_idx=exp['start_idx'],
            end_idx=exp['end_idx'],
            threshold=exp['threshold'],
            number=args.number,
            quiet=args.quiet
        )
        
        if result:
            all_results.append(result)
        
        # Save intermediate results every 5 experiments
        if i % 5 == 0:
            with open(os.path.join(args.output_dir, 'intermediate_results.json'), 'w') as f:
                json.dump(all_results, f, indent=2)
    
    total_time = time.time() - total_time
    print(f"\nCompleted {len(all_results)}/{len(experiments)} experiments in {total_time/60:.1f} minutes")
    
    # Generate summary report
    if all_results:
        create_summary_report(all_results, args.output_dir)
    else:
        print("No successful experiments to summarize!")


if __name__ == '__main__':
    main()
