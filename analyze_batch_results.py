#!/usr/bin/env python3
"""
Batch Results Analyzer for JailGuard Systematic Testing
Analyzes and summarizes results from comprehensive batch testing
"""

import json
import sys
from pathlib import Path
import pandas as pd
import argparse
from datetime import datetime

def load_batch_results(batch_summary_file):
    """Load batch results from summary file"""
    try:
        with open(batch_summary_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading batch results: {e}")
        return None

def analyze_detection_performance(batch_results):
    """Analyze detection performance across all datasets"""
    safe_results = []
    unsafe_results = []
    
    for test_name, result in batch_results["results"].items():
        if result["test_result"]["status"] != "success":
            continue
            
        config = result["config"]
        detailed = result.get("detailed_results", {})
        
        analysis = {
            "test_name": test_name,
            "description": config["description"],
            "expected_toxicity": config["expected_toxicity"],
            "total_samples": detailed.get("total_samples", 0),
            "successful_tests": detailed.get("successful_tests", 0),
            "attacks_detected": detailed.get("attacks_detected", 0),
            "detection_rate": detailed.get("detection_rate", 0.0),
            "avg_divergence": detailed.get("avg_divergence", 0.0),
            "max_divergence": detailed.get("max_divergence", 0.0),
            "avg_processing_time": detailed.get("avg_processing_time", 0.0)
        }
        
        if config["expected_toxicity"] == 0:
            safe_results.append(analysis)
        else:
            unsafe_results.append(analysis)
    
    return safe_results, unsafe_results

def calculate_overall_metrics(safe_results, unsafe_results):
    """Calculate overall performance metrics"""
    # Safe datasets (should have low detection rates - false positives)
    safe_total_samples = sum(r["total_samples"] for r in safe_results)
    safe_false_positives = sum(r["attacks_detected"] for r in safe_results)
    safe_false_positive_rate = safe_false_positives / safe_total_samples if safe_total_samples > 0 else 0
    
    # Unsafe datasets (should have high detection rates - true positives)
    unsafe_total_samples = sum(r["total_samples"] for r in unsafe_results)
    unsafe_true_positives = sum(r["attacks_detected"] for r in unsafe_results)
    unsafe_detection_rate = unsafe_true_positives / unsafe_total_samples if unsafe_total_samples > 0 else 0
    
    # Overall accuracy
    total_samples = safe_total_samples + unsafe_total_samples
    correct_predictions = (safe_total_samples - safe_false_positives) + unsafe_true_positives
    overall_accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    
    return {
        "safe_datasets": {
            "total_samples": safe_total_samples,
            "false_positives": safe_false_positives,
            "false_positive_rate": safe_false_positive_rate,
            "specificity": 1 - safe_false_positive_rate
        },
        "unsafe_datasets": {
            "total_samples": unsafe_total_samples,
            "true_positives": unsafe_true_positives,
            "detection_rate": unsafe_detection_rate,
            "sensitivity": unsafe_detection_rate
        },
        "overall": {
            "total_samples": total_samples,
            "accuracy": overall_accuracy,
            "correct_predictions": correct_predictions
        }
    }

def print_detailed_analysis(batch_results):
    """Print comprehensive analysis of batch results"""
    print(f"{'='*80}")
    print(f"JAILGUARD SYSTEMATIC TEST ANALYSIS")
    print(f"{'='*80}")
    
    batch_info = batch_results["batch_info"]
    print(f"📅 Test Date: {batch_info['timestamp']}")
    print(f"⏱️  Total Duration: {batch_info['total_duration_seconds']/60:.1f} minutes")
    print(f"🔧 Configuration: {batch_info['configuration']}")
    
    safe_results, unsafe_results = analyze_detection_performance(batch_results)
    overall_metrics = calculate_overall_metrics(safe_results, unsafe_results)
    
    print(f"\n{'='*60}")
    print(f"OVERALL PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    
    print(f"📊 Total Samples Tested: {overall_metrics['overall']['total_samples']}")
    print(f"🎯 Overall Accuracy: {overall_metrics['overall']['accuracy']:.1%}")
    print(f"✅ Correct Predictions: {overall_metrics['overall']['correct_predictions']}")
    
    print(f"\n🛡️  SAFE DATASETS (Should NOT be detected as attacks):")
    print(f"   Total samples: {overall_metrics['safe_datasets']['total_samples']}")
    print(f"   False positives: {overall_metrics['safe_datasets']['false_positives']}")
    print(f"   False positive rate: {overall_metrics['safe_datasets']['false_positive_rate']:.1%}")
    print(f"   Specificity: {overall_metrics['safe_datasets']['specificity']:.1%}")
    
    print(f"\n⚠️  UNSAFE DATASETS (Should BE detected as attacks):")
    print(f"   Total samples: {overall_metrics['unsafe_datasets']['total_samples']}")
    print(f"   True positives: {overall_metrics['unsafe_datasets']['true_positives']}")
    print(f"   Detection rate: {overall_metrics['unsafe_datasets']['detection_rate']:.1%}")
    print(f"   Sensitivity: {overall_metrics['unsafe_datasets']['sensitivity']:.1%}")
    
    print(f"\n{'='*60}")
    print(f"DETAILED RESULTS BY DATASET")
    print(f"{'='*60}")
    
    print(f"\n🛡️  SAFE DATASETS:")
    print(f"{'Dataset':<25} {'Samples':<8} {'Detected':<9} {'Rate':<8} {'Avg Div':<10} {'Max Div':<10}")
    print(f"{'-'*75}")
    for result in safe_results:
        print(f"{result['test_name']:<25} {result['total_samples']:<8} "
              f"{result['attacks_detected']:<9} {result['detection_rate']:<8.1%} "
              f"{result['avg_divergence']:<10.4f} {result['max_divergence']:<10.4f}")
    
    print(f"\n⚠️  UNSAFE DATASETS:")
    print(f"{'Dataset':<25} {'Samples':<8} {'Detected':<9} {'Rate':<8} {'Avg Div':<10} {'Max Div':<10}")
    print(f"{'-'*75}")
    for result in unsafe_results:
        print(f"{result['test_name']:<25} {result['total_samples']:<8} "
              f"{result['attacks_detected']:<9} {result['detection_rate']:<8.1%} "
              f"{result['avg_divergence']:<10.4f} {result['max_divergence']:<10.4f}")
    
    # Performance analysis
    print(f"\n{'='*60}")
    print(f"PERFORMANCE ANALYSIS")
    print(f"{'='*60}")
    
    all_results = safe_results + unsafe_results
    if all_results:
        avg_processing_time = sum(r["avg_processing_time"] for r in all_results) / len(all_results)
        total_processing_time = sum(r["avg_processing_time"] * r["total_samples"] for r in all_results)
        
        print(f"⏱️  Average processing time per sample: {avg_processing_time:.2f}s")
        print(f"⏱️  Total processing time: {total_processing_time/60:.1f} minutes")
        
        # Divergence analysis
        avg_divergences = [r["avg_divergence"] for r in all_results if r["avg_divergence"] > 0]
        max_divergences = [r["max_divergence"] for r in all_results if r["max_divergence"] > 0]
        
        if avg_divergences:
            print(f"📈 Average divergence across datasets: {sum(avg_divergences)/len(avg_divergences):.4f}")
            print(f"📈 Maximum divergence observed: {max(max_divergences):.4f}")
    
    # Failed tests
    failed_tests = [name for name, result in batch_results["results"].items() 
                   if result["test_result"]["status"] != "success"]
    
    if failed_tests:
        print(f"\n{'='*60}")
        print(f"FAILED TESTS")
        print(f"{'='*60}")
        for test_name in failed_tests:
            result = batch_results["results"][test_name]
            print(f"❌ {test_name}: {result['test_result']['status']}")
            if "error" in result["test_result"]:
                print(f"   Error: {result['test_result']['error'][:100]}...")

def export_to_csv(batch_results, output_file):
    """Export results to CSV for further analysis"""
    safe_results, unsafe_results = analyze_detection_performance(batch_results)
    all_results = safe_results + unsafe_results
    
    if not all_results:
        print("No results to export")
        return
    
    df = pd.DataFrame(all_results)
    df.to_csv(output_file, index=False)
    print(f"📊 Results exported to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Analyze JailGuard batch test results")
    parser.add_argument("batch_summary", help="Path to batch_summary.json file")
    parser.add_argument("--export-csv", help="Export results to CSV file")
    parser.add_argument("--quiet", action="store_true", help="Only show summary metrics")
    
    args = parser.parse_args()
    
    batch_results = load_batch_results(args.batch_summary)
    if not batch_results:
        sys.exit(1)
    
    if not args.quiet:
        print_detailed_analysis(batch_results)
    else:
        safe_results, unsafe_results = analyze_detection_performance(batch_results)
        overall_metrics = calculate_overall_metrics(safe_results, unsafe_results)
        print(f"Overall Accuracy: {overall_metrics['overall']['accuracy']:.1%}")
        print(f"Detection Rate (Sensitivity): {overall_metrics['unsafe_datasets']['detection_rate']:.1%}")
        print(f"False Positive Rate: {overall_metrics['safe_datasets']['false_positive_rate']:.1%}")
    
    if args.export_csv:
        export_to_csv(batch_results, args.export_csv)

if __name__ == "__main__":
    main()
