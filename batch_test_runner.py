#!/usr/bin/env python3
"""
Batch Test Runner for JailGuard Systematic Testing
Runs comprehensive tests on all datasets from the 1800 sample test configuration
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime
from pathlib import Path
import argparse

# Test configuration matching the 1800 sample test set
TEST_CONFIGURATION = {
    "safe_datasets": {
        "XSTest_safe": {
            "dataset_name": "XSTest",
            "max_samples": 250,
            "expected_toxicity": 0,
            "filter_toxicity": 0,
            "description": "XSTest safe subset"
        },
        "FigTxt_safe": {
            "dataset_name": "FigTxt",
            "max_samples": 300,
            "expected_toxicity": 0,
            "filter_toxicity": 0,
            "description": "FigTxt safe subset"
        },
        "VQAv2": {
            "dataset_name": "VQAv2",
            "max_samples": 350,
            "expected_toxicity": 0,
            "filter_toxicity": None,  # VQAv2 is already all safe
            "description": "VQAv2 visual question answering"
        }
    },
    "unsafe_datasets": {
        "XSTest_unsafe": {
            "dataset_name": "XSTest",
            "max_samples": 200,
            "expected_toxicity": 1,
            "filter_toxicity": 1,
            "description": "XSTest unsafe subset"
        },
        "FigTxt_unsafe": {
            "dataset_name": "FigTxt",
            "max_samples": 350,
            "expected_toxicity": 1,
            "filter_toxicity": 1,
            "description": "FigTxt unsafe subset"
        },
        "VAE": {
            "dataset_name": "Adversarial_Images",
            "max_samples": 200,
            "expected_toxicity": 1,
            "filter_toxicity": None,  # VAE is already all unsafe
            "description": "VAE adversarial images"
        },
        "JailbreakV_figstep": {
            "dataset_name": "JailBreakV_figstep",
            "max_samples": 150,
            "expected_toxicity": 1,
            "filter_toxicity": None,  # JailbreakV is already all unsafe
            "description": "JailbreakV-28K figstep attack"
        }
    }
}

def run_systematic_test(dataset_name, max_samples, num_variants, output_dir, threshold=0.025, mutator="PL", filter_toxicity=None):
    """Run systematic test for a single dataset"""
    print(f"\n{'='*60}")
    print(f"Starting test: {dataset_name}")
    print(f"Max samples: {max_samples}, Variants: {num_variants}")
    if filter_toxicity is not None:
        toxicity_label = "safe" if filter_toxicity == 0 else "unsafe"
        print(f"Toxicity filter: {toxicity_label} samples only")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, "systematic_test_jailguard.py",
        "--dataset", dataset_name,
        "--max-samples", str(max_samples),
        "--num-variants", str(num_variants),
        "--output-dir", output_dir,
        "--threshold", str(threshold),
        "--mutator", mutator
    ]

    # Add toxicity filter if specified
    if filter_toxicity is not None:
        cmd.extend(["--filter-toxicity", str(filter_toxicity)])
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
            # No timeout - let it run as long as needed for large datasets
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ {dataset_name} completed successfully in {duration:.1f}s")
            return {
                "status": "success",
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        else:
            print(f"❌ {dataset_name} failed with return code {result.returncode}")
            print(f"Error: {result.stderr}")
            return {
                "status": "failed",
                "duration": duration,
                "error": result.stderr,
                "stdout": result.stdout
            }
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {dataset_name} timed out after 1 hour")
        return {
            "status": "timeout",
            "duration": 3600,
            "error": "Test timed out after 1 hour"
        }
    except Exception as e:
        print(f"💥 {dataset_name} crashed: {e}")
        return {
            "status": "crashed",
            "duration": time.time() - start_time,
            "error": str(e)
        }

def load_results_summary(results_file):
    """Load and parse results from a systematic test output"""
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        report = data.get('report', {})
        return {
            "total_samples": report.get('test_summary', {}).get('total_samples', 0),
            "successful_tests": report.get('test_summary', {}).get('successful_tests', 0),
            "attacks_detected": report.get('detection_results', {}).get('attacks_detected', 0),
            "detection_rate": report.get('detection_results', {}).get('detection_rate', 0.0),
            "avg_divergence": report.get('divergence_analysis', {}).get('avg_divergence', 0.0),
            "max_divergence": report.get('divergence_analysis', {}).get('max_divergence', 0.0),
            "avg_processing_time": report.get('performance_metrics', {}).get('avg_processing_time_seconds', 0.0)
        }
    except Exception as e:
        print(f"Warning: Could not load results from {results_file}: {e}")
        return {}

def main():
    parser = argparse.ArgumentParser(description="Batch test runner for JailGuard systematic testing")
    parser.add_argument("--num-variants", type=int, default=3, help="Number of variants per sample (default: 3)")
    parser.add_argument("--threshold", type=float, default=0.025, help="Detection threshold (default: 0.025)")
    parser.add_argument("--mutator", type=str, default="PL", help="Mutator type (default: PL)")
    parser.add_argument("--output-base-dir", type=str, default="batch_test_results", help="Base output directory")
    parser.add_argument("--safe-only", action="store_true", help="Test only safe datasets")
    parser.add_argument("--unsafe-only", action="store_true", help="Test only unsafe datasets")
    parser.add_argument("--resume", action="store_true", help="Resume from existing results")
    
    args = parser.parse_args()
    
    # Create base output directory
    base_output_dir = Path(args.output_base_dir)
    base_output_dir.mkdir(exist_ok=True)
    
    # Handle resume vs new run
    if args.resume:
        # Find the most recent batch directory to resume from
        existing_batch_dirs = list(base_output_dir.glob("batch_*"))
        if existing_batch_dirs:
            # Sort by name (which includes timestamp) and get the most recent
            batch_dir = sorted(existing_batch_dirs)[-1]
            print(f"🔄 Resuming from existing batch: {batch_dir.name}")
        else:
            print("⚠️  No existing batch found to resume from, starting new batch")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_dir = base_output_dir / f"batch_{timestamp}"
            batch_dir.mkdir(exist_ok=True)
    else:
        # Create new timestamped directory for fresh run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = base_output_dir / f"batch_{timestamp}"
        batch_dir.mkdir(exist_ok=True)
    
    print(f"🚀 Starting batch test run at {datetime.now()}")
    print(f"📁 Results will be saved to: {batch_dir}")
    print(f"🔧 Configuration: {args.num_variants} variants, threshold={args.threshold}, mutator={args.mutator}")
    
    # Determine which datasets to test
    datasets_to_test = {}
    if args.safe_only:
        datasets_to_test.update(TEST_CONFIGURATION["safe_datasets"])
    elif args.unsafe_only:
        datasets_to_test.update(TEST_CONFIGURATION["unsafe_datasets"])
    else:
        datasets_to_test.update(TEST_CONFIGURATION["safe_datasets"])
        datasets_to_test.update(TEST_CONFIGURATION["unsafe_datasets"])
    
    print(f"📊 Testing {len(datasets_to_test)} datasets:")
    for name, config in datasets_to_test.items():
        print(f"  - {name}: {config['description']} ({config['max_samples']} samples)")
    
    # Run tests
    batch_results = {}
    total_start_time = time.time()
    
    for test_name, config in datasets_to_test.items():
        output_dir = batch_dir / f"test_{test_name}"
        
        # Check if we should skip or resume this dataset
        if args.resume and output_dir.exists():
            # Check checkpoint to see actual progress
            checkpoint_files = list(output_dir.glob("checkpoint_*.json"))
            if checkpoint_files:
                try:
                    import json
                    with open(checkpoint_files[0], 'r') as f:
                        checkpoint_data = json.load(f)

                    expected_samples = checkpoint_data.get('config', {}).get('max_samples', 0)
                    completed_samples = len(checkpoint_data.get('results', []))

                    if completed_samples >= expected_samples:
                        print(f"⏭️  Skipping {test_name} (completed: {completed_samples}/{expected_samples})")
                        continue
                    else:
                        print(f"🔄 Resuming {test_name} (completed: {completed_samples}/{expected_samples})")
                        # Continue to run the test - it will resume from checkpoint
                except Exception as e:
                    print(f"⚠️  Could not read checkpoint for {test_name}: {e}")
                    print(f"🔄 Running {test_name} anyway")
            else:
                print(f"⚠️  No checkpoint found for {test_name}, but directory exists")
                print(f"🔄 Running {test_name} anyway")
        
        result = run_systematic_test(
            dataset_name=config["dataset_name"],
            max_samples=config["max_samples"],
            num_variants=args.num_variants,
            output_dir=str(output_dir),
            threshold=args.threshold,
            mutator=args.mutator,
            filter_toxicity=config.get("filter_toxicity")
        )
        
        batch_results[test_name] = {
            "config": config,
            "test_result": result,
            "timestamp": datetime.now().isoformat()
        }
        
        # Try to load detailed results if test succeeded
        if result["status"] == "success":
            results_files = list(output_dir.glob("results_*.json"))
            if results_files:
                detailed_results = load_results_summary(results_files[0])
                batch_results[test_name]["detailed_results"] = detailed_results
    
    total_duration = time.time() - total_start_time
    
    # Save batch results summary
    batch_summary = {
        "batch_info": {
            "timestamp": timestamp,
            "total_duration_seconds": total_duration,
            "configuration": {
                "num_variants": args.num_variants,
                "threshold": args.threshold,
                "mutator": args.mutator
            }
        },
        "results": batch_results
    }
    
    summary_file = batch_dir / "batch_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(batch_summary, f, indent=2)
    
    # Print final summary
    print(f"\n{'='*80}")
    print(f"🏁 BATCH TEST COMPLETED")
    print(f"{'='*80}")
    print(f"⏱️  Total duration: {total_duration/60:.1f} minutes")
    print(f"📁 Results saved to: {batch_dir}")
    
    successful_tests = sum(1 for r in batch_results.values() if r["test_result"]["status"] == "success")
    failed_tests = len(batch_results) - successful_tests
    
    print(f"✅ Successful tests: {successful_tests}/{len(batch_results)}")
    if failed_tests > 0:
        print(f"❌ Failed tests: {failed_tests}")
        for name, result in batch_results.items():
            if result["test_result"]["status"] != "success":
                print(f"   - {name}: {result['test_result']['status']}")
    
    print(f"\n📊 Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()
