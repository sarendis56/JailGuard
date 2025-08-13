#!/usr/bin/env python3
"""
JailGuard Results Analysis Script
=================================

This script provides comprehensive analysis and reporting capabilities for JailGuard test results.
It can analyze individual test runs, compare multiple runs, and generate detailed reports.

Features:
- Load and analyze test results from JSON/CSV files
- Generate statistical summaries and visualizations
- Compare performance across different datasets and configurations
- Export analysis reports in multiple formats

Usage:
    python analyze_results.py --results results_file.json
    python analyze_results.py --compare results1.json results2.json
    python analyze_results.py --directory results_dir/ --generate-report
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ResultsAnalyzer:
    """Analyzer for JailGuard test results"""
    
    def __init__(self):
        self.results_data = []
        self.datasets = {}
        
    def load_results(self, results_file: str) -> Dict[str, Any]:
        """Load results from JSON file"""
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        self.results_data.append(data)
        dataset_name = data.get('config', {}).get('dataset_name', 'unknown')
        self.datasets[dataset_name] = data
        
        print(f"Loaded results for dataset: {dataset_name}")
        print(f"  Total samples: {len(data.get('results', []))}")
        
        return data
    
    def load_results_directory(self, directory: str):
        """Load all result files from a directory"""
        results_dir = Path(directory)
        json_files = list(results_dir.glob("results_*.json"))
        
        if not json_files:
            print(f"No result files found in {directory}")
            return
        
        print(f"Found {len(json_files)} result files")
        for file in json_files:
            try:
                self.load_results(file)
            except Exception as e:
                print(f"Error loading {file}: {e}")
    
    def analyze_single_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Analyze results for a single dataset"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not loaded")
        
        data = self.datasets[dataset_name]
        results = data.get('results', [])
        config = data.get('config', {})
        
        if not results:
            return {"error": "No results to analyze"}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(results)
        
        # Basic statistics
        total_samples = len(df)
        successful_tests = len(df[df['error_message'].isna()])
        failed_tests = len(df[df['error_message'].notna()])
        attacks_detected = len(df[df['detection_result'] == True])
        
        # Performance metrics
        processing_times = df[df['processing_time'] > 0]['processing_time']
        divergences = df[df['max_divergence'] > 0]['max_divergence']
        
        # Toxicity analysis (if available)
        toxicity_analysis = {}
        if 'sample_data' in df.columns:
            # Extract toxicity labels from sample data
            toxicity_labels = []
            for sample_data in df['sample_data']:
                if isinstance(sample_data, dict):
                    toxicity_labels.append(sample_data.get('toxicity', 'unknown'))
                else:
                    toxicity_labels.append('unknown')
            
            df['toxicity_label'] = toxicity_labels
            
            # Analyze detection performance by toxicity
            if 'toxicity_label' in df.columns:
                toxicity_groups = df.groupby('toxicity_label')
                for label, group in toxicity_groups:
                    if label != 'unknown':
                        detected = len(group[group['detection_result'] == True])
                        total = len(group)
                        toxicity_analysis[f'toxicity_{label}'] = {
                            'total_samples': total,
                            'detected': detected,
                            'detection_rate': detected / total if total > 0 else 0
                        }
        
        # Divergence distribution analysis
        divergence_stats = {}
        if len(divergences) > 0:
            divergence_stats = {
                'mean': float(divergences.mean()),
                'median': float(divergences.median()),
                'std': float(divergences.std()),
                'min': float(divergences.min()),
                'max': float(divergences.max()),
                'percentiles': {
                    '25th': float(divergences.quantile(0.25)),
                    '75th': float(divergences.quantile(0.75)),
                    '90th': float(divergences.quantile(0.90)),
                    '95th': float(divergences.quantile(0.95))
                }
            }
        
        # Error analysis
        error_analysis = {}
        if failed_tests > 0:
            error_messages = df[df['error_message'].notna()]['error_message']
            error_counts = error_messages.value_counts().to_dict()
            error_analysis = {
                'total_errors': failed_tests,
                'unique_errors': len(error_counts),
                'most_common_errors': dict(list(error_counts.items())[:5])
            }
        
        analysis = {
            'dataset_info': {
                'name': dataset_name,
                'total_samples': total_samples,
                'successful_tests': successful_tests,
                'failed_tests': failed_tests,
                'success_rate': successful_tests / total_samples if total_samples > 0 else 0
            },
            'detection_performance': {
                'attacks_detected': attacks_detected,
                'detection_rate': attacks_detected / successful_tests if successful_tests > 0 else 0,
                'threshold_used': config.get('threshold', 'unknown')
            },
            'toxicity_analysis': toxicity_analysis,
            'performance_metrics': {
                'avg_processing_time': float(processing_times.mean()) if len(processing_times) > 0 else 0,
                'total_processing_time': float(processing_times.sum()) if len(processing_times) > 0 else 0,
                'processing_time_std': float(processing_times.std()) if len(processing_times) > 0 else 0
            },
            'divergence_analysis': divergence_stats,
            'error_analysis': error_analysis,
            'configuration': config,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        return analysis
    
    def compare_datasets(self, dataset_names: List[str]) -> Dict[str, Any]:
        """Compare results across multiple datasets"""
        if not all(name in self.datasets for name in dataset_names):
            missing = [name for name in dataset_names if name not in self.datasets]
            raise ValueError(f"Datasets not loaded: {missing}")
        
        comparison = {
            'datasets_compared': dataset_names,
            'comparison_metrics': {},
            'summary_table': []
        }
        
        for dataset_name in dataset_names:
            analysis = self.analyze_single_dataset(dataset_name)
            
            summary_row = {
                'dataset': dataset_name,
                'total_samples': analysis['dataset_info']['total_samples'],
                'success_rate': analysis['dataset_info']['success_rate'],
                'detection_rate': analysis['detection_performance']['detection_rate'],
                'avg_processing_time': analysis['performance_metrics']['avg_processing_time'],
                'avg_divergence': analysis['divergence_analysis'].get('mean', 0),
                'threshold': analysis['detection_performance']['threshold_used']
            }
            comparison['summary_table'].append(summary_row)
        
        # Create comparison DataFrame
        df_comparison = pd.DataFrame(comparison['summary_table'])
        
        # Add ranking
        df_comparison['detection_rate_rank'] = df_comparison['detection_rate'].rank(ascending=False)
        df_comparison['processing_time_rank'] = df_comparison['avg_processing_time'].rank(ascending=True)
        
        comparison['summary_dataframe'] = df_comparison
        comparison['comparison_timestamp'] = datetime.now().isoformat()
        
        return comparison
    
    def generate_visualizations(self, output_dir: str = "analysis_plots"):
        """Generate visualization plots for the loaded results"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        plt.style.use('seaborn-v0_8')
        
        for dataset_name, data in self.datasets.items():
            results = data.get('results', [])
            if not results:
                continue
            
            df = pd.DataFrame(results)
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'JailGuard Analysis: {dataset_name}', fontsize=16)
            
            # Plot 1: Detection Results
            detection_counts = df['detection_result'].value_counts()
            axes[0, 0].pie(detection_counts.values, labels=['Benign', 'Attack'], autopct='%1.1f%%')
            axes[0, 0].set_title('Detection Results Distribution')
            
            # Plot 2: Divergence Distribution
            divergences = df[df['max_divergence'] > 0]['max_divergence']
            if len(divergences) > 0:
                axes[0, 1].hist(divergences, bins=30, alpha=0.7, edgecolor='black')
                axes[0, 1].axvline(data.get('config', {}).get('threshold', 0.025), 
                                 color='red', linestyle='--', label='Threshold')
                axes[0, 1].set_xlabel('Max Divergence')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].set_title('Divergence Distribution')
                axes[0, 1].legend()
            
            # Plot 3: Processing Time Distribution
            times = df[df['processing_time'] > 0]['processing_time']
            if len(times) > 0:
                axes[1, 0].hist(times, bins=20, alpha=0.7, edgecolor='black')
                axes[1, 0].set_xlabel('Processing Time (seconds)')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].set_title('Processing Time Distribution')
            
            # Plot 4: Success Rate by Sample
            success_rate = df['error_message'].isna().rolling(window=50, min_periods=1).mean()
            axes[1, 1].plot(success_rate.index, success_rate.values)
            axes[1, 1].set_xlabel('Sample Index')
            axes[1, 1].set_ylabel('Success Rate')
            axes[1, 1].set_title('Success Rate Over Time')
            axes[1, 1].set_ylim(0, 1)
            
            plt.tight_layout()
            
            # Save plot
            plot_file = output_path / f"{dataset_name}_analysis.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Visualization saved: {plot_file}")
    
    def export_analysis_report(self, output_file: str, format: str = "json"):
        """Export comprehensive analysis report"""
        if not self.datasets:
            print("No datasets loaded for analysis")
            return
        
        report = {
            'analysis_summary': {
                'total_datasets': len(self.datasets),
                'datasets_analyzed': list(self.datasets.keys()),
                'analysis_timestamp': datetime.now().isoformat()
            },
            'individual_analyses': {},
            'comparison': None
        }
        
        # Analyze each dataset individually
        for dataset_name in self.datasets.keys():
            report['individual_analyses'][dataset_name] = self.analyze_single_dataset(dataset_name)
        
        # Compare datasets if multiple are loaded
        if len(self.datasets) > 1:
            report['comparison'] = self.compare_datasets(list(self.datasets.keys()))
        
        # Export report
        if format.lower() == "json":
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        elif format.lower() == "csv":
            # Export summary table as CSV
            if report['comparison']:
                df = pd.DataFrame(report['comparison']['summary_table'])
                df.to_csv(output_file, index=False)
        
        print(f"Analysis report exported: {output_file}")
        return report


def main():
    parser = argparse.ArgumentParser(
        description="JailGuard Results Analysis Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single result file
  python analyze_results.py --results results_JailBreakV_figstep_20240101_120000.json

  # Analyze all results in a directory
  python analyze_results.py --directory systematic_test_results/

  # Compare specific datasets
  python analyze_results.py --compare results1.json results2.json

  # Generate comprehensive report with visualizations
  python analyze_results.py --directory results/ --generate-report --visualizations
        """
    )

    # Input options
    parser.add_argument('--results', type=str, help='Single results file to analyze')
    parser.add_argument('--directory', type=str, help='Directory containing result files')
    parser.add_argument('--compare', nargs='+', help='Multiple result files to compare')

    # Analysis options
    parser.add_argument('--generate-report', action='store_true', help='Generate comprehensive analysis report')
    parser.add_argument('--visualizations', action='store_true', help='Generate visualization plots')
    parser.add_argument('--output-format', choices=['json', 'csv'], default='json', help='Output format for reports')

    # Output options
    parser.add_argument('--output-dir', type=str, default='analysis_output', help='Output directory for reports and plots')
    parser.add_argument('--report-name', type=str, help='Custom name for the analysis report')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Initialize analyzer
    analyzer = ResultsAnalyzer()

    # Load results
    if args.results:
        analyzer.load_results(args.results)
    elif args.directory:
        analyzer.load_results_directory(args.directory)
    elif args.compare:
        for file in args.compare:
            analyzer.load_results(file)
    else:
        print("Error: No input specified. Use --results, --directory, or --compare")
        return

    if not analyzer.datasets:
        print("No valid results loaded.")
        return

    print(f"\nLoaded {len(analyzer.datasets)} dataset(s): {list(analyzer.datasets.keys())}")

    # Generate analysis
    if len(analyzer.datasets) == 1:
        # Single dataset analysis
        dataset_name = list(analyzer.datasets.keys())[0]
        analysis = analyzer.analyze_single_dataset(dataset_name)

        print(f"\n{'='*60}")
        print(f"ANALYSIS SUMMARY: {dataset_name}")
        print(f"{'='*60}")

        # Print key metrics
        info = analysis['dataset_info']
        detection = analysis['detection_performance']
        performance = analysis['performance_metrics']
        divergence = analysis['divergence_analysis']

        print(f"Dataset: {info['name']}")
        print(f"Total samples: {info['total_samples']}")
        print(f"Successful tests: {info['successful_tests']} ({info['success_rate']:.1%})")
        print(f"Failed tests: {info['failed_tests']}")
        print()
        print(f"Attacks detected: {detection['attacks_detected']}")
        print(f"Detection rate: {detection['detection_rate']:.1%}")
        print(f"Threshold used: {detection['threshold_used']}")
        print()
        print(f"Average processing time: {performance['avg_processing_time']:.2f}s")
        print(f"Total processing time: {performance['total_processing_time']:.2f}s")

        if divergence:
            print()
            print(f"Divergence statistics:")
            print(f"  Mean: {divergence['mean']:.4f}")
            print(f"  Median: {divergence['median']:.4f}")
            print(f"  Max: {divergence['max']:.4f}")
            print(f"  95th percentile: {divergence['percentiles']['95th']:.4f}")

        # Toxicity analysis
        if analysis['toxicity_analysis']:
            print()
            print("Detection by toxicity label:")
            for label, stats in analysis['toxicity_analysis'].items():
                print(f"  {label}: {stats['detected']}/{stats['total_samples']} ({stats['detection_rate']:.1%})")

        # Error analysis
        if analysis['error_analysis']:
            print()
            print(f"Errors encountered: {analysis['error_analysis']['total_errors']}")
            if analysis['error_analysis']['most_common_errors']:
                print("Most common errors:")
                for error, count in analysis['error_analysis']['most_common_errors'].items():
                    print(f"  {error}: {count}")

    else:
        # Multi-dataset comparison
        dataset_names = list(analyzer.datasets.keys())
        comparison = analyzer.compare_datasets(dataset_names)

        print(f"\n{'='*80}")
        print(f"COMPARISON SUMMARY: {len(dataset_names)} datasets")
        print(f"{'='*80}")

        # Print comparison table
        df = comparison['summary_dataframe']
        print("\nDataset Comparison:")
        print(df.to_string(index=False, float_format='%.3f'))

        # Print rankings
        print(f"\nTop datasets by detection rate:")
        top_detection = df.nlargest(3, 'detection_rate')[['dataset', 'detection_rate']]
        for _, row in top_detection.iterrows():
            print(f"  {row['dataset']}: {row['detection_rate']:.1%}")

        print(f"\nFastest datasets by processing time:")
        fastest = df.nsmallest(3, 'avg_processing_time')[['dataset', 'avg_processing_time']]
        for _, row in fastest.iterrows():
            print(f"  {row['dataset']}: {row['avg_processing_time']:.2f}s")

    # Generate comprehensive report
    if args.generate_report:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.report_name:
            report_file = output_dir / f"{args.report_name}.{args.output_format}"
        else:
            report_file = output_dir / f"jailguard_analysis_{timestamp}.{args.output_format}"

        analyzer.export_analysis_report(str(report_file), args.output_format)

    # Generate visualizations
    if args.visualizations:
        plot_dir = output_dir / "plots"
        analyzer.generate_visualizations(str(plot_dir))

    print(f"\nAnalysis complete. Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
