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

# Import model training/testing (only test will be used when models exist)
from train_model import train_model
from test_models import test_model


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
    
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Training Phase (actually skipping if model exists)
    if not test_only:
        print("\n📚 Phase 1: Checking Models / Training")
        print("-" * 40)
        for i, model_name in enumerate(models_to_compare, 1):
            try:
                print(f"\n[{i}/{len(models_to_compare)}] {model_name.upper()}...")

                # Look for any .pth files containing the model name
                matching_models = [f for f in model_dir.glob("*.pth") if model_name.lower() in f.name.lower()]


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
    print("\n📊 Phase 3: Analysis & Comparison")
    print("-" * 40)
    
    successful_models = [m for m in models_to_compare if 'error' not in results['test_results'].get(m, {})]
    
    if successful_models:
        # Create comparison summary
        comparison_data = []
        for model_name in successful_models:
            test_res = results['test_results'][model_name]
            train_res = results['training_results'].get(model_name, {})
            
            comparison_data.append({
                'Model': model_name.upper(),
                'Mean IoU': test_res.get('mean_iou', 0),
                'Pixel Accuracy': test_res.get('pixel_accuracy', 0),
                'mAP@50': test_res.get('map_50', 0),
                'mAP@75': test_res.get('map_75', 0),
                'Mean Dice': test_res.get('mean_dice', 0),
                'Training IoU': train_res.get('best_iou_score'),
                'Training Time (min)': (train_res.get('training_time_seconds') or 0) / 60,
                'Test Time (sec)': test_res.get('test_time_seconds', 0)
            })
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Mean IoU', ascending=False)
        
        results['comparison_summary'] = {
            'best_model_by_iou': df.iloc[0]['Model'].lower(),
            'best_mean_iou': df.iloc[0]['Mean IoU'],
            'ranking': df[['Model', 'Mean IoU', 'Pixel Accuracy', 'mAP@50']].to_dict('records')
        }
        
        # Print results
        print("\n🏆 Model Comparison Results:")
        print(df.to_string(index=False, float_format='%.4f'))
        
        # Save results
        if save_results:
            save_comparison_results(results, df)
            create_comparison_plots(df, results['experiment_info']['timestamp'])
    
    else:
        print("❌ No models completed successfully for comparison.")
    
    print(f"\n✅ Comparison completed! Results saved to output/model_comparison/")
    return results


def save_comparison_results(results, df):
    """Save comparison results to files."""
    output_dir = Path("output/model_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = results['experiment_info']['timestamp']
    
    json_file = output_dir / f"comparison_results_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    csv_file = output_dir / f"model_comparison_summary_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    
    print(f"📁 Results saved: {csv_file}, {json_file}")


def create_comparison_plots(df, timestamp):
    """Create visualization plots for model comparison."""
    output_dir = Path("output/model_comparison")
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Mean IoU
    fig, ax = plt.subplots(figsize=(8,6))
    ax.bar(df['Model'], df['Mean IoU'], color='skyblue')
    ax.set_title('Mean IoU by Model')
    ax.set_ylabel('Mean IoU')
    plt.tight_layout()
    plt.savefig(output_dir / f"performance_comparison_{timestamp}.png", dpi=300)
    plt.close()
    print(f"📊 Plot saved: performance_comparison_{timestamp}.png")


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
        best_model = results['comparison_summary']['best_model_by_iou']
        best_score = results['comparison_summary']['best_mean_iou']
        print(f"\n🏆 Winner: {best_model.upper()} with Mean IoU: {best_score:.4f}")
    else:
        print("\n❌ Comparison could not be completed.")
