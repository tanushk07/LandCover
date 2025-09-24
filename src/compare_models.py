"""
Automated model comparison script for different segmentation architectures.
This script trains multiple models and compares their performance.

Usage: 
    python compare_models.py --models unet deeplabv3 linknet
    python compare_models.py --models all  # trains all available models
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
from collections import defaultdict

# Import model training function
from train_model import train_model
from test_models import test_model


def get_available_models():
    """Get list of available model configurations."""
    from pathlib import Path
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
    
    # Training Phase
    if not test_only:
        print("\n📚 Phase 1: Training Models")
        print("-" * 40)
        
        for i, model_name in enumerate(models_to_compare, 1):
            try:
                print(f"\n[{i}/{len(models_to_compare)}] Training {model_name.upper()}...")
                start_time = time.time()
                
                max_score, best_epoch = train_model(model_name)
                training_time = time.time() - start_time
                
                results['training_results'][model_name] = {
                    'best_iou_score': max_score,
                    'best_epoch': best_epoch,
                    'training_time_seconds': training_time,
                    'training_time_formatted': f"{training_time//3600:.0f}h {(training_time%3600)//60:.0f}m {training_time%60:.0f}s"
                }
                
                print(f"✅ {model_name.upper()} completed - IoU: {max_score:.4f}, Time: {results['training_results'][model_name]['training_time_formatted']}")
                
            except Exception as e:
                print(f"❌ {model_name.upper()} training failed: {str(e)}")
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
            
            print(f"✅ {model_name.upper()} tested - mIoU: {test_metrics.get('mean_iou', 'N/A'):.4f}")
            
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
                'Training IoU': train_res.get('best_iou_score', 0),
                'Training Time (min)': train_res.get('training_time_seconds', 0) / 60,
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
    # Create output directory
    output_dir = Path("output/model_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = results['experiment_info']['timestamp']
    
    # Save JSON results
    json_file = output_dir / f"comparison_results_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save CSV summary
    csv_file = output_dir / f"model_comparison_summary_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    
    # Save detailed results CSV
    detailed_file = output_dir / f"detailed_results_{timestamp}.csv"
    detailed_data = []
    
    for model_name in results['test_results']:
        if 'error' not in results['test_results'][model_name]:
            test_res = results['test_results'][model_name]
            train_res = results['training_results'].get(model_name, {})
            
            detailed_data.append({
                'Model': model_name.upper(),
                'Architecture': model_name,
                'Mean IoU': test_res.get('mean_iou', 0),
                'Pixel Accuracy': test_res.get('pixel_accuracy', 0),
                'Mean Pixel Accuracy': test_res.get('mean_pixel_accuracy', 0),
                'mAP@50': test_res.get('map_50', 0),
                'mAP@75': test_res.get('map_75', 0),
                'Mean Dice': test_res.get('mean_dice', 0),
                'Frequency Weighted IoU': test_res.get('frequency_weighted_iou', 0),
                'Training Best IoU': train_res.get('best_iou_score', 0),
                'Training Best Epoch': train_res.get('best_epoch', 0),
                'Training Time (min)': train_res.get('training_time_seconds', 0) / 60,
                'Test Time (sec)': test_res.get('test_time_seconds', 0)
            })
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_df.to_csv(detailed_file, index=False)
    
    print(f"📁 Results saved:")
    print(f"   - Summary: {csv_file}")
    print(f"   - Detailed: {detailed_file}")
    print(f"   - Full JSON: {json_file}")


def create_comparison_plots(df, timestamp):
    """Create visualization plots for model comparison."""
    output_dir = Path("output/model_comparison")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Performance Metrics Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Architecture Performance Comparison', fontsize=16, fontweight='bold')
    
    # Mean IoU
    axes[0, 0].bar(df['Model'], df['Mean IoU'], color='skyblue', alpha=0.8)
    axes[0, 0].set_title('Mean IoU Score')
    axes[0, 0].set_ylabel('IoU Score')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Pixel Accuracy
    axes[0, 1].bar(df['Model'], df['Pixel Accuracy'], color='lightgreen', alpha=0.8)
    axes[0, 1].set_title('Pixel Accuracy')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # mAP@50
    axes[1, 0].bar(df['Model'], df['mAP@50'], color='salmon', alpha=0.8)
    axes[1, 0].set_title('Mean Average Precision @50')
    axes[1, 0].set_ylabel('mAP@50')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Mean Dice
    axes[1, 1].bar(df['Model'], df['Mean Dice'], color='gold', alpha=0.8)
    axes[1, 1].set_title('Mean Dice Coefficient')
    axes[1, 1].set_ylabel('Dice Score')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"performance_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Training vs Test Performance
    if 'Training IoU' in df.columns:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        x = range(len(df))
        width = 0.35
        
        ax.bar([i - width/2 for i in x], df['Training IoU'], width, label='Training IoU', alpha=0.8)
        ax.bar([i + width/2 for i in x], df['Mean IoU'], width, label='Test IoU', alpha=0.8)
        
        ax.set_xlabel('Model Architecture')
        ax.set_ylabel('IoU Score')
        ax.set_title('Training vs Test IoU Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(df['Model'], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"train_vs_test_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Performance vs Time Trade-off
    if 'Training Time (min)' in df.columns:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        scatter = ax.scatter(df['Training Time (min)'], df['Mean IoU'], 
                           s=100, alpha=0.7, c=df.index, cmap='viridis')
        
        for i, model in enumerate(df['Model']):
            ax.annotate(model, (df['Training Time (min)'].iloc[i], df['Mean IoU'].iloc[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax.set_xlabel('Training Time (minutes)')
        ax.set_ylabel('Mean IoU Score')
        ax.set_title('Performance vs Training Time Trade-off')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"performance_vs_time_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"📊 Plots saved:")
    print(f"   - Performance comparison: performance_comparison_{timestamp}.png")
    print(f"   - Train vs Test: train_vs_test_{timestamp}.png")
    print(f"   - Performance vs Time: performance_vs_time_{timestamp}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare different segmentation model architectures')
    parser.add_argument('--models', nargs='+', default=['unet', 'deeplabv3', 'linknet'],
                       help='Models to compare. Use "all" for all available models.')
    parser.add_argument('--test-only', action='store_true',
                       help='Only test existing models without training new ones')
    
    args = parser.parse_args()
    
    # Handle 'all' models option
    if 'all' in args.models:
        models_to_compare = get_available_models()
    else:
        available_models = get_available_models()
        models_to_compare = [m for m in args.models if m in available_models]
        
        if len(models_to_compare) != len(args.models):
            invalid_models = [m for m in args.models if m not in available_models]
            print(f"⚠️  Invalid models ignored: {invalid_models}")
            print(f"✅ Available models: {available_models}")
    
    if not models_to_compare:
        print("❌ No valid models specified!")
        exit(1)
    
    print(f"🎯 Models to compare: {[m.upper() for m in models_to_compare]}")
    print(f"🔄 Test only mode: {args.test_only}")
    
    # Run comparison
    results = compare_models(models_to_compare, test_only=args.test_only)
    
    # Print final summary
    if results['comparison_summary']:
        best_model = results['comparison_summary']['best_model_by_iou']
        best_score = results['comparison_summary']['best_mean_iou']
        print(f"\n🏆 Winner: {best_model.upper()} with Mean IoU: {best_score:.4f}")
    else:
        print("\n❌ Comparison could not be completed due to errors.")
