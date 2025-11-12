"""
Automated model comparison script for different segmentation architectures.
This script can skip training if pre-trained models already exist.

Usage: 
    python compare_models.py --models unet deeplabv3 linknet
    python compare_models.py --models all  # use all available models
    python compare_models.py --models all --test-only  # only test pre-trained models
"""

import os
import json
import time
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import numpy as np

# Import model training/testing (only test will be used when models exist)
from train_model import train_model
from test_models import test_model


# Metric weights for overall score
METRIC_WEIGHTS = {
    'mean_iou': 1.0,
    'mean_dice': 0.9,
    'pixel_accuracy': 0.7,
    'mean_pixel_accuracy': 0.7,
    'frequency_weighted_iou': 0.4,
    'map_50': 0.3,
    'map_75': 0.3
}
METRIC_COLUMN_MAP = {
    'mean_iou': 'Mean IoU',
    'mean_dice': 'Mean Dice',
    'pixel_accuracy': 'Pixel Accuracy',
    'mean_pixel_accuracy': 'Mean Pixel Accuracy',
    'frequency_weighted_iou': 'Frequency Weighted IoU',
    'map_50': 'mAP@50',
    'map_75': 'mAP@75'
}

def get_available_models():
    """Get list of available model configurations."""
    config_dir = Path(__file__).parent.parent / "config" / "models"
    available_configs = [f.stem.replace('_config', '') for f in config_dir.glob('*_config.yaml')]
    return available_configs


def compare_models(models_to_compare, test_only=False, save_results=True):
    """
    Compare multiple model architectures.
    
    Args:
        models_to_compare (list): List of model names to compare
        test_only (bool): If True, only test existing models without training
        save_results (bool): Whether to save comparison results
    
    Returns:
        dict: Comparison results
    """
    
    print("🚀 Starting Model Architecture Comparison")
    print("=" * 60)
    
    results = {
        'training_results': {},
        'test_results': {},
        'comparison_summary': {},
        'experiment_info': {
            'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S'),
            'models_compared': models_to_compare,
            'test_only': test_only
        }
    }
    
    model_dir = Path(__file__).resolve().parent.parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Training Phase (actually skipping if model exists)
    if not test_only:
        print("\n📚 Phase 1: Checking Models / Training")
        print("-" * 40)
        for i, model_name in enumerate(models_to_compare, 1):
            try:
                print(f"\n[{i}/{len(models_to_compare)}] {model_name.upper()}...")

                # Look for any .pth files containing the model name
                normalized_model_name = model_name.lower().replace("-", "_").replace(" ", "")
                matching_models = [
                    f for f in model_dir.glob("*.pth")
                    if normalized_model_name in f.name.lower().replace("-", "_").replace(" ", "")
                ]

                if matching_models:
                    print(f"⏭️  {model_name.upper()} skipped — found trained model: {matching_models[0].name}")
                    results['training_results'][model_name] = {
                        'best_iou_score': None,
                        'best_epoch': None,
                        'training_time_seconds': None,
                        'training_time_formatted': None,
                        'status': 'skipped_existing_model'
                    }
                else:
                    print(f"🔨 Training {model_name.upper()} (no trained model found)...")
                    start_time = time.time()
                    
                    max_score, best_epoch = train_model(model_name)
                    training_time = time.time() - start_time
                    
                    results['training_results'][model_name] = {
                        'best_iou_score': max_score,
                        'best_epoch': best_epoch,
                        'training_time_seconds': training_time,
                        'training_time_formatted': f"{training_time//3600:.0f}h {(training_time%3600)//60:.0f}m {training_time%60:.0f}s",
                        'status': 'trained'
                    }
                    
                    print(f"✅ {model_name.upper()} completed - IoU: {max_score:.4f}, Time: {results['training_results'][model_name]['training_time_formatted']}")

            except Exception as e:
                print(f"❌ {model_name.upper()} failed in training phase: {str(e)}")
                results['training_results'][model_name] = {
                    'error': str(e),
                    'status': 'failed'
                }
    
    # Testing Phase
    print("\n🧪 Phase 2: Testing Models")
    print("-" * 40)
    
    for i, model_name in enumerate(models_to_compare, 1):
        try:
            print(f"\n[{i}/{len(models_to_compare)}] Testing {model_name.upper()}...")
            start_time = time.time()
            
            test_metrics = test_model(model_name)
            test_time = time.time() - start_time
            
            results['test_results'][model_name] = {
                **test_metrics,
                'test_time_seconds': test_time,
                'test_time_formatted': f"{test_time//60:.0f}m {test_time%60:.0f}s"
            }
            
            print(f"✅ {model_name.upper()} tested - mIoU: {test_metrics.get('mean_iou', 0):.4f}")
            
        except Exception as e:
            print(f"❌ {model_name.upper()} testing failed: {str(e)}")
            results['test_results'][model_name] = {
                'error': str(e),
                'status': 'failed'
            }
    
    # Analysis Phase
    # Analysis Phase
# Analysis Phase
    print("\n📊 Phase 3: Analysis & Comparison")
    print("-" * 40)

    successful_models = [m for m in models_to_compare if 'error' not in results['test_results'].get(m, {})]

    print(f"✅ Successfully tested models: {successful_models}")
    print(f"📊 Total models for comparison: {len(successful_models)}")

    if successful_models:
        comparison_data = []
        
        for model_name in successful_models:
            print(f"📝 Processing {model_name.upper()} for comparison...")
            
            test_res = results['test_results'][model_name]
            train_res = results['training_results'].get(model_name, {})
            
            # Validate that test_res has actual metrics
            miou = test_res.get('mean_iou')
            if miou is None or (isinstance(miou, float) and (np.isnan(miou))):
                print(f"⚠️  Skipping {model_name} - no metrics found")
                continue

            
            # Collect metrics
            metrics = {
                'Mean IoU': test_res.get('mean_iou', 0),
                'Pixel Accuracy': test_res.get('pixel_accuracy', 0),
                'Mean Pixel Accuracy': test_res.get('mean_pixel_accuracy', 0),
                'Frequency Weighted IoU': test_res.get('frequency_weighted_iou', 0),
                'Mean Dice': test_res.get('mean_dice', 0),
                'mAP@50': test_res.get('map_50', 0),
                'mAP@75': test_res.get('map_75', 0),
            }
            
            # Print warning for invalid values (INSIDE the loop)
            for metric_name, value in metrics.items():
                if value < -1 or value > 1:  
                    print(f"⚠️  WARNING: {model_name} has invalid {metric_name}: {value}")
            
            # Append to comparison data (INSIDE the loop)
            comparison_data.append({
                'Model': model_name.upper(),
                **metrics,
                'Training IoU': train_res.get('best_iou_score'),
                'Training Time (min)': (train_res.get('training_time_seconds') or 0) / 60,
                'Test Time (sec)': test_res.get('test_time_seconds', 0)
            })
            
            print(f"✅ {model_name.upper()} added to comparison (mIoU: {metrics['Mean IoU']:.4f})")
        
        if not comparison_data:
            print("❌ No valid data collected for comparison!")
            return results

        df = pd.DataFrame(comparison_data)
        print(f"\n📊 DataFrame created with {len(df)} models")

        # Determine best model per metric
        best_per_metric = {}
        for metric_key in METRIC_WEIGHTS.keys():
            col_name = METRIC_COLUMN_MAP.get(metric_key, metric_key)
            if col_name in df.columns:
                best_row = df.loc[df[col_name].idxmax()]
                best_per_metric[metric_key] = {
                    'model': best_row['Model'],
                    'score': best_row[col_name]
                }

        # Compute weighted overall score
        def weighted_score(row):
            score = 0
            weight_sum = 0
            for metric_key, weight in METRIC_WEIGHTS.items():
                col_name = METRIC_COLUMN_MAP.get(metric_key, metric_key)
                if col_name in row and pd.notna(row[col_name]):
                    value = row[col_name]
                    if -1 <= value <= 1:  # Basic sanity check
                        score += value * weight
                        weight_sum += weight
                    else:
                        print(f"⚠️  {col_name} has invalid value {value} for {row.get('Model', 'unknown')}")
            return score / weight_sum if weight_sum > 0 else 0

        df['Weighted Score'] = df.apply(weighted_score, axis=1)
        best_overall_row = df.loc[df['Weighted Score'].idxmax()]

        results['comparison_summary'] = {
            'best_per_metric': best_per_metric,
            'best_model_overall': {
                'model': best_overall_row['Model'],
                'weighted_score': best_overall_row['Weighted Score']
            },
            'full_ranking': df.sort_values('Weighted Score', ascending=False).to_dict('records')
        }

        # Print results
        print("\n🏆 Model Comparison Results:")
        print(df.to_string(index=False, float_format='%.4f'))

        # Save results
        if save_results:
            save_comparison_results(results, df)
            create_comparison_plots(df, results['experiment_info']['timestamp'])

            # Bar chart for Weighted Score
            plt.figure(figsize=(10, 6))
            plt.bar(df['Model'], df['Weighted Score'], color='skyblue', edgecolor='navy', alpha=0.7)
            plt.title("Final Weighted Scores (Overall Model Performance)", fontsize=14, fontweight='bold')
            plt.ylabel("Weighted Score", fontsize=12)
            plt.xlabel("Model", fontsize=12)
            plt.xticks(rotation=30, ha="right")
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"output/model_comparison/weighted_scores_{results['experiment_info']['timestamp']}.png", dpi=300)
            plt.close()
            print(f"📊 Weighted scores plot saved")
    else:
        print("❌ No models completed successfully for comparison.")

    print(f"\n✅ Comparison completed! Results saved to output/model_comparison/")
    return results

def make_json_safe(obj):
    """Convert numpy types to native Python types for json.dump compatibility."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # fallback for pandas types
    try:
        import pandas as pd
        if isinstance(obj, (pd.Timestamp,)):
            return str(obj)
    except Exception:
        pass
    raise TypeError(f"Type {type(obj)} not serializable")

def save_comparison_results(results, df):
    """Save comparison results to files."""
    output_dir = Path("output/model_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = results['experiment_info']['timestamp']
    
    json_file = output_dir / f"comparison_results_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2,default=make_json_safe)
    
    csv_file = output_dir / f"model_comparison_summary_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    
    print(f"📁 Results saved: {csv_file}, {json_file}")


def create_comparison_plots(df, timestamp):
    """Create visualization plots for all metrics."""
    output_dir = Path("output/model_comparison")
    plt.style.use('default')
    sns.set_palette("husl")
    
    metrics_to_plot = [
        'Mean IoU', 'Pixel Accuracy', 'Mean Pixel Accuracy', 
        'Frequency Weighted IoU', 'Mean Dice', 'mAP@50', 'mAP@75'
    ]
    
    fig, ax = plt.subplots(figsize=(10,6))
    width = 0.1  # width of bars
    x = range(len(df['Model']))
    
    for i, metric in enumerate(metrics_to_plot):
        if metric in df.columns:
            ax.bar([p + i*width for p in x], df[metric], width=width, label=metric)

    ax.set_xticks([p + width*len(metrics_to_plot)/2 for p in x])
    ax.set_xticklabels(df['Model'])
    ax.set_ylabel('Metric Value')
    ax.set_title('Segmentation Model Performance Comparison')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"performance_comparison_all_metrics_{timestamp}.png", dpi=300)
    plt.close()
    print(f"📊 Plot saved: performance_comparison_all_metrics_{timestamp}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare different segmentation model architectures')
    parser.add_argument('--models', nargs='+', default=['unet', 'deeplabv3', 'linknet'],
                       help='Models to compare. Use "all" for all available models.')
    parser.add_argument('--test-only', action='store_true',
                       help='Only test existing models without training new ones')
    
    args = parser.parse_args()
    
    if 'all' in args.models:
        models_to_compare = get_available_models()
    else:
        available_models = get_available_models()
        models_to_compare = [m for m in args.models if m in available_models]
        if len(models_to_compare) != len(args.models):
            invalid = [m for m in args.models if m not in available_models]
            print(f"⚠️ Invalid models ignored: {invalid}")
            print(f"✅ Available models: {available_models}")
    
    if not models_to_compare:
        print("❌ No valid models specified!")
        exit(1)
    
    print(f"🎯 Models to compare: {[m.upper() for m in models_to_compare]}")
    print(f"🔄 Test only mode: {args.test_only}")
    
    results = compare_models(models_to_compare, test_only=args.test_only)
    
    if results['comparison_summary']:
        best_model = results['comparison_summary']['best_model_overall']['model']
        best_score = results['comparison_summary']['best_model_overall']['weighted_score']
        print(f"\n🏆 Winner: {best_model.upper()} with Weighted Score: {best_score:.4f}")

    else:
        print("\n❌ Comparison could not be completed.")
