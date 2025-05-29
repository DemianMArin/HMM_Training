"""
analysis_utils.py
Enhanced utility functions for analyzing and verifying codevector data.
Supports both LPC and MFCC feature types.
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Union
import pandas as pd
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import MaxNLocator
import sys

from codevector_classes import RawData, RawDataMFCC, CentroidData, CentroidDataMFCC, DataStorage


def analyze_frame_calculations_lpc(frames: List[RawData]) -> Dict:
    """Analyze the quality of LPC frame calculations."""
    analysis = {
        'total_frames': len(frames),
        'frames_with_valid_autocorr': 0,
        'frames_with_valid_lpc': 0,
        'frames_with_valid_lsf': 0,
        'frames_with_issues': [],
        'raw_sample_stats': {
            'min_length': float('inf'),
            'max_length': 0,
            'avg_length': 0,
            'lengths': []
        }
    }
    
    for i, frame in enumerate(frames):
        # Check raw sample length
        sample_len = len(frame.raw_samples)
        analysis['raw_sample_stats']['lengths'].append(sample_len)
        analysis['raw_sample_stats']['min_length'] = min(analysis['raw_sample_stats']['min_length'], sample_len)
        analysis['raw_sample_stats']['max_length'] = max(analysis['raw_sample_stats']['max_length'], sample_len)
        
        # Check calculations
        has_valid_autocorr = not np.allclose(frame.autocorrelation_vector, 0) or sample_len == 0
        has_valid_lpc = not np.allclose(frame.lpc_vector, 0) or sample_len <= 12
        has_valid_lsf = not np.allclose(frame.lsf_vector, 0) or np.allclose(frame.lpc_vector, 0)
        
        if has_valid_autocorr:
            analysis['frames_with_valid_autocorr'] += 1
        if has_valid_lpc:
            analysis['frames_with_valid_lpc'] += 1
        if has_valid_lsf:
            analysis['frames_with_valid_lsf'] += 1
        
        # Identify problematic frames
        issues = []
        if sample_len > 0 and np.allclose(frame.autocorrelation_vector, 0):
            issues.append('zero_autocorr')
        if sample_len > 12 and np.allclose(frame.lpc_vector, 0):
            issues.append('zero_lpc')
        if not np.allclose(frame.lpc_vector, 0) and np.allclose(frame.lsf_vector, 0):
            issues.append('zero_lsf')
        
        if issues:
            analysis['frames_with_issues'].append({
                'frame_index': i,
                'recording': frame.recording,
                'frame_number': frame.frame_number,
                'sample_length': sample_len,
                'issues': issues
            })
    
    # Calculate average length
    if analysis['raw_sample_stats']['lengths']:
        analysis['raw_sample_stats']['avg_length'] = np.mean(analysis['raw_sample_stats']['lengths'])
    
    return analysis


def analyze_frame_calculations_mfcc(frames: List[RawDataMFCC]) -> Dict:
    """Analyze the quality of MFCC frame calculations."""
    analysis = {
        'total_frames': len(frames),
        'frames_with_valid_mfcc': 0,
        'frames_with_issues': [],
        'raw_sample_stats': {
            'min_length': float('inf'),
            'max_length': 0,
            'avg_length': 0,
            'lengths': []
        },
        'mfcc_stats': {
            'min_magnitude': float('inf'),
            'max_magnitude': 0,
            'avg_magnitude': 0,
            'magnitudes': []
        }
    }
    
    for i, frame in enumerate(frames):
        # Check raw sample length
        sample_len = len(frame.raw_samples)
        analysis['raw_sample_stats']['lengths'].append(sample_len)
        analysis['raw_sample_stats']['min_length'] = min(analysis['raw_sample_stats']['min_length'], sample_len)
        analysis['raw_sample_stats']['max_length'] = max(analysis['raw_sample_stats']['max_length'], sample_len)
        
        # Check MFCC calculations
        mfcc_magnitude = np.linalg.norm(frame.mfcc)
        analysis['mfcc_stats']['magnitudes'].append(mfcc_magnitude)
        analysis['mfcc_stats']['min_magnitude'] = min(analysis['mfcc_stats']['min_magnitude'], mfcc_magnitude)
        analysis['mfcc_stats']['max_magnitude'] = max(analysis['mfcc_stats']['max_magnitude'], mfcc_magnitude)
        
        has_valid_mfcc = not np.allclose(frame.mfcc, 0) or sample_len == 0
        
        if has_valid_mfcc:
            analysis['frames_with_valid_mfcc'] += 1
        
        # Identify problematic frames
        issues = []
        if sample_len > 0 and np.allclose(frame.mfcc, 0):
            issues.append('zero_mfcc')
        if len(frame.mfcc) != 13:
            issues.append(f'wrong_mfcc_size_{len(frame.mfcc)}')
        
        if issues:
            analysis['frames_with_issues'].append({
                'frame_index': i,
                'recording': frame.recording,
                'frame_number': frame.frame_number,
                'sample_length': sample_len,
                'mfcc_size': len(frame.mfcc),
                'mfcc_magnitude': mfcc_magnitude,
                'issues': issues
            })
    
    # Calculate averages
    if analysis['raw_sample_stats']['lengths']:
        analysis['raw_sample_stats']['avg_length'] = np.mean(analysis['raw_sample_stats']['lengths'])
    if analysis['mfcc_stats']['magnitudes']:
        analysis['mfcc_stats']['avg_magnitude'] = np.mean(analysis['mfcc_stats']['magnitudes'])
    
    return analysis


def analyze_centroid_assignments(frames: Union[List[RawData], List[RawDataMFCC]]) -> Dict:
    """Analyze how frames are assigned to centroids."""
    analysis = {
        'total_frames': len(frames),
        'frames_with_assignments': 0,
        'unique_centroids': set(),
        'unique_generations': set(),
        'centroid_distribution': {},
        'generation_distribution': {},
        'recording_distribution': {}
    }
    
    for frame in frames:
        # Count frames with assignments
        if hasattr(frame, 'parent_centroid_id') and frame.parent_centroid_id is not None:
            analysis['frames_with_assignments'] += 1
            analysis['unique_centroids'].add(frame.parent_centroid_id)
            
            # Count frames per centroid
            if frame.parent_centroid_id not in analysis['centroid_distribution']:
                analysis['centroid_distribution'][frame.parent_centroid_id] = 0
            analysis['centroid_distribution'][frame.parent_centroid_id] += 1
        
        # Count generations
        if hasattr(frame, 'generation') and frame.generation is not None:
            analysis['unique_generations'].add(frame.generation)
            
            # Count frames per generation
            if frame.generation not in analysis['generation_distribution']:
                analysis['generation_distribution'][frame.generation] = 0
            analysis['generation_distribution'][frame.generation] += 1
        
        # Count frames per recording
        if frame.recording not in analysis['recording_distribution']:
            analysis['recording_distribution'][frame.recording] = 0
        analysis['recording_distribution'][frame.recording] += 1
    
    # Convert sets to sorted lists for easier viewing
    analysis['unique_centroids'] = sorted(list(analysis['unique_centroids']))
    analysis['unique_generations'] = sorted(list(analysis['unique_generations']))
    
    return analysis


def print_frame_analysis_lpc(analysis: Dict):
    """Print a formatted analysis of LPC frame data."""
    print("=" * 60)
    print("LPC FRAME ANALYSIS REPORT")
    print("=" * 60)
    
    print(f"Total frames analyzed: {analysis['total_frames']}")
    print(f"Frames with valid autocorrelation: {analysis['frames_with_valid_autocorr']}")
    print(f"Frames with valid LPC: {analysis['frames_with_valid_lpc']}")
    print(f"Frames with valid LSF: {analysis['frames_with_valid_lsf']}")
    
    print(f"\nRaw sample statistics:")
    stats = analysis['raw_sample_stats']
    print(f"  Min length: {stats['min_length']} samples")
    print(f"  Max length: {stats['max_length']} samples")
    print(f"  Average length: {stats['avg_length']:.1f} samples")
    
    if analysis['frames_with_issues']:
        print(f"\nFrames with issues: {len(analysis['frames_with_issues'])}")
        for issue_info in analysis['frames_with_issues'][:5]:  # Show first 5
            print(f"  Frame {issue_info['frame_index']} ({issue_info['recording']}, frame {issue_info['frame_number']})")
            print(f"    Sample length: {issue_info['sample_length']}, Issues: {issue_info['issues']}")
        if len(analysis['frames_with_issues']) > 5:
            print(f"  ... and {len(analysis['frames_with_issues']) - 5} more")
    else:
        print(f"\n✓ No calculation issues found!")


def print_frame_analysis_mfcc(analysis: Dict):
    """Print a formatted analysis of MFCC frame data."""
    print("=" * 60)
    print("MFCC FRAME ANALYSIS REPORT")
    print("=" * 60)
    
    print(f"Total frames analyzed: {analysis['total_frames']}")
    print(f"Frames with valid MFCC: {analysis['frames_with_valid_mfcc']}")
    
    print(f"\nRaw sample statistics:")
    stats = analysis['raw_sample_stats']
    print(f"  Min length: {stats['min_length']} samples")
    print(f"  Max length: {stats['max_length']} samples")
    print(f"  Average length: {stats['avg_length']:.1f} samples")
    
    print(f"\nMFCC statistics:")
    mfcc_stats = analysis['mfcc_stats']
    print(f"  Min magnitude: {mfcc_stats['min_magnitude']:.3f}")
    print(f"  Max magnitude: {mfcc_stats['max_magnitude']:.3f}")
    print(f"  Average magnitude: {mfcc_stats['avg_magnitude']:.3f}")
    
    if analysis['frames_with_issues']:
        print(f"\nFrames with issues: {len(analysis['frames_with_issues'])}")
        for issue_info in analysis['frames_with_issues'][:5]:  # Show first 5
            print(f"  Frame {issue_info['frame_index']} ({issue_info['recording']}, frame {issue_info['frame_number']})")
            print(f"    Sample length: {issue_info['sample_length']}, MFCC size: {issue_info['mfcc_size']}")
            print(f"    MFCC magnitude: {issue_info['mfcc_magnitude']:.3f}, Issues: {issue_info['issues']}")
        if len(analysis['frames_with_issues']) > 5:
            print(f"  ... and {len(analysis['frames_with_issues']) - 5} more")
    else:
        print(f"\n✓ No calculation issues found!")


def print_assignment_analysis(analysis: Dict):
    """Print a formatted analysis of centroid assignments."""
    print("=" * 60)
    print("CENTROID ASSIGNMENT ANALYSIS")
    print("=" * 60)
    
    print(f"Total frames: {analysis['total_frames']}")
    print(f"Frames with centroid assignments: {analysis['frames_with_assignments']}")
    print(f"Unique centroids used: {len(analysis['unique_centroids'])}")
    print(f"Unique generations: {len(analysis['unique_generations'])}")
    
    if analysis['unique_generations']:
        print(f"Generations used: {analysis['unique_generations']}")
    
    if analysis['centroid_distribution']:
        assignments = list(analysis['centroid_distribution'].values())
        print(f"\nCentroid usage distribution:")
        print(f"  Min frames per centroid: {min(assignments)}")
        print(f"  Max frames per centroid: {max(assignments)}")
        print(f"  Average frames per centroid: {np.mean(assignments):.1f}")
        print(f"  Std dev: {np.std(assignments):.1f}")
    
    if analysis['recording_distribution']:
        recordings = list(analysis['recording_distribution'].values())
        print(f"\nRecording distribution:")
        print(f"  Total recordings: {len(analysis['recording_distribution'])}")
        print(f"  Min frames per recording: {min(recordings)}")
        print(f"  Max frames per recording: {max(recordings)}")
        print(f"  Average frames per recording: {np.mean(recordings):.1f}")


def plot_centroid_distribution(analysis: Dict, save_path: str = None):
    """Plot the distribution of frames across centroids."""
    if not analysis['centroid_distribution']:
        print("No centroid distribution data to plot")
        return
    
    centroids = list(analysis['centroid_distribution'].keys())
    counts = list(analysis['centroid_distribution'].values())
    
    plt.figure(figsize=(12, 6))

    # Plot histogram of frame counts per centroid
    plt.subplot(1, 2, 1)
    plt.hist(counts, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Frames per Centroid')
    plt.ylabel('Number of Centroids')
    plt.title('Distribution of Frames per Centroid')
    plt.grid(True, alpha=0.3)

    # Set maximum number of ticks
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=4))
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=1))

    # Plot centroid usage over centroid IDs
    plt.subplot(1, 2, 2)
    plt.plot(centroids, counts, 'o', alpha=0.7, markersize=3)
    plt.xlabel('Centroid ID')
    plt.ylabel('Number of Frames')
    plt.title('Frame Count by Centroid ID')
    plt.grid(True, alpha=0.3)

    # Set maximum number of ticks
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=4))
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=10))

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def plot_mfcc_analysis(analysis: Dict, save_path: str = None):
    """Plot MFCC-specific analysis."""
    if not analysis['mfcc_stats']['magnitudes']:
        print("No MFCC data to plot")
        return
    
    magnitudes = analysis['mfcc_stats']['magnitudes']
    
    plt.figure(figsize=(12, 4))
    
    # Plot histogram of MFCC magnitudes
    plt.subplot(1, 2, 1)
    plt.hist(magnitudes, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('MFCC Vector Magnitude')
    plt.ylabel('Number of Frames')
    plt.title('Distribution of MFCC Magnitudes')
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=8))
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=8))
    
    # Plot MFCC magnitudes over frame index (sample)
    plt.subplot(1, 2, 2)
    sample_indices = range(0, len(magnitudes), max(1, len(magnitudes)//1000))  # Sample for large datasets
    sampled_magnitudes = [magnitudes[i] for i in sample_indices]
    plt.plot(sample_indices, sampled_magnitudes, alpha=0.7, linewidth=0.5)
    plt.xlabel('Frame Index (sampled)')
    plt.ylabel('MFCC Magnitude')
    plt.title('MFCC Magnitude Over Time')
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=8))
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"MFCC analysis plot saved to: {save_path}")
    else:
        plt.show()


def verify_codevector_data_lpc(base_dir: str = "Data"):
    """Complete verification of LPC codevector data."""
    print("LPC CODEVECTOR DATA VERIFICATION")
    print("=" * 60)
    
    storage = DataStorage()
    base_path = Path(base_dir)
    codevector_dir = base_path / "CodeVector"
    
    # Check if codevector directory exists
    if not codevector_dir.exists():
        print(f"❌ CodeVector directory not found: {codevector_dir}")
        return
    
    # Load original training frames
    original_frames_file = codevector_dir / "codevector_frames.pkl"
    if original_frames_file.exists():
        print(f"Loading original training frames from: {original_frames_file}")
        original_frames = storage.load_data_binary(str(original_frames_file))
        
        print(f"\nAnalyzing original training frames...")
        calc_analysis = analyze_frame_calculations_lpc(original_frames)
        print_frame_analysis_lpc(calc_analysis)
        
        assign_analysis = analyze_centroid_assignments(original_frames)
        print_assignment_analysis(assign_analysis)
    else:
        print(f"❌ Original training frames not found: {original_frames_file}")
        return
    
    # Load updated training frames
    updated_frames_file = codevector_dir / "codevector_frames_updated.pkl"
    if updated_frames_file.exists():
        print(f"\nLoading updated training frames from: {updated_frames_file}")
        updated_frames = storage.load_data_binary(str(updated_frames_file))
        
        print(f"\nAnalyzing updated training frames...")
        updated_calc_analysis = analyze_frame_calculations_lpc(updated_frames)
        print_frame_analysis_lpc(updated_calc_analysis)
        
        updated_assign_analysis = analyze_centroid_assignments(updated_frames)
        print_assignment_analysis(updated_assign_analysis)
        
        # Plot distribution
        plot_path = codevector_dir / "centroid_distribution_lpc.png"
        plot_centroid_distribution(updated_assign_analysis, str(plot_path))
        
    else:
        print(f"⚠️ Updated training frames not found: {updated_frames_file}")
        print("This is expected if codevector creation hasn't been run yet.")
    
    # Load codevector
    codevector_file = codevector_dir / "codevector.pkl"
    if codevector_file.exists():
        print(f"\nLoading codevector from: {codevector_file}")
        centroids = storage.load_data_binary(str(codevector_file))
        print(f"✓ Codevector loaded with {len(centroids)} centroids")
        
        # Analyze centroids
        lsf_magnitudes = [np.linalg.norm(c.lsf) for c in centroids]
        print(f"LSF magnitude statistics:")
        print(f"  Min: {min(lsf_magnitudes):.3f}")
        print(f"  Max: {max(lsf_magnitudes):.3f}")
        print(f"  Mean: {np.mean(lsf_magnitudes):.3f}")
        print(f"  Std: {np.std(lsf_magnitudes):.3f}")
    else:
        print(f"❌ Codevector not found: {codevector_file}")
    
    # Load training summary
    summary_file = codevector_dir / "training_summary.json"
    if summary_file.exists():
        print(f"\nLoading training summary from: {summary_file}")
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        print(f"Training Summary:")
        print(f"  Total frames: {summary['total_frames']}")
        print(f"  Max generation: {summary['max_generation']}")
        print(f"  Centroids with assignments: {len(summary['centroid_assignments'])}")
        
        if summary['centroid_assignments']:
            assignments = list(summary['centroid_assignments'].values())
            print(f"  Frame distribution:")
            print(f"    Min: {min(assignments)}")
            print(f"    Max: {max(assignments)}")
            print(f"    Mean: {np.mean(assignments):.1f}")
    else:
        print(f"⚠️ Training summary not found: {summary_file}")
    
    print("\n" + "=" * 60)
    print("LPC VERIFICATION COMPLETE")
    print("=" * 60)


def verify_codevector_data_mfcc(base_dir: str = "Data"):
    """Complete verification of MFCC codevector data."""
    print("MFCC CODEVECTOR DATA VERIFICATION")
    print("=" * 60)
    
    storage = DataStorage()
    base_path = Path(base_dir)
    # codevector_dir = base_path / "CodeVectorMFCC"
    codevector_dir = base_path / "CodeVector"

    # Check if codevector directory exists
    if not codevector_dir.exists():
        print(f"❌ CodeVectorMFCC directory not found: {codevector_dir}")
        return
    
    # Load original training frames
    original_frames_file = codevector_dir / "codevector_frames.pkl"
    if original_frames_file.exists():
        print(f"Loading original MFCC training frames from: {original_frames_file}")
        original_frames = storage.load_data_binary(str(original_frames_file))
        
        print(f"\nAnalyzing original MFCC training frames...")
        calc_analysis = analyze_frame_calculations_mfcc(original_frames)
        print_frame_analysis_mfcc(calc_analysis)
        
        assign_analysis = analyze_centroid_assignments(original_frames)
        print_assignment_analysis(assign_analysis)
        
        # Plot MFCC-specific analysis
        mfcc_plot_path = codevector_dir / "mfcc_analysis.png"
        plot_mfcc_analysis(calc_analysis, str(mfcc_plot_path))
    else:
        print(f"❌ Original MFCC training frames not found: {original_frames_file}")
        return
    
    # Load updated training frames
    updated_frames_file = codevector_dir / "codevector_frames_updated.pkl"
    if updated_frames_file.exists():
        print(f"\nLoading updated MFCC training frames from: {updated_frames_file}")
        updated_frames = storage.load_data_binary(str(updated_frames_file))
        
        print(f"\nAnalyzing updated MFCC training frames...")
        updated_calc_analysis = analyze_frame_calculations_mfcc(updated_frames)
        print_frame_analysis_mfcc(updated_calc_analysis)
        
        updated_assign_analysis = analyze_centroid_assignments(updated_frames)
        print_assignment_analysis(updated_assign_analysis)
        
        # Plot distribution
        plot_path = codevector_dir / "centroid_distribution_mfcc.png"
        plot_centroid_distribution(updated_assign_analysis, str(plot_path))
        
        # Plot updated MFCC analysis
        mfcc_plot_path = codevector_dir / "mfcc_analysis_updated.png"
        plot_mfcc_analysis(updated_calc_analysis, str(mfcc_plot_path))
        
    else:
        print(f"⚠️ Updated MFCC training frames not found: {updated_frames_file}")
        print("This is expected if MFCC codevector creation hasn't been run yet.")
    
    # Load MFCC codevector
    codevector_file = codevector_dir / "codevector.pkl"
    if codevector_file.exists():
        print(f"\nLoading MFCC codevector from: {codevector_file}")
        centroids = storage.load_data_binary(str(codevector_file))
        print(f"✓ MFCC Codevector loaded with {len(centroids)} centroids")
        
        # Analyze MFCC centroids
        mfcc_magnitudes = [np.linalg.norm(c.mfcc) for c in centroids]
        print(f"MFCC magnitude statistics:")
        print(f"  Min: {min(mfcc_magnitudes):.3f}")
        print(f"  Max: {max(mfcc_magnitudes):.3f}")
        print(f"  Mean: {np.mean(mfcc_magnitudes):.3f}")
        print(f"  Std: {np.std(mfcc_magnitudes):.3f}")
    else:
        print(f"❌ MFCC Codevector not found: {codevector_file}")
    
    # Load training summary
    summary_file = codevector_dir / "training_summary.json"
    if summary_file.exists():
        print(f"\nLoading MFCC training summary from: {summary_file}")
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        print(f"MFCC Training Summary:")
        print(f"  Total frames: {summary['total_frames']}")
        print(f"  Max generation: {summary['max_generation']}")
        print(f"  Centroids with assignments: {len(summary['centroid_assignments'])}")
        
        if summary['centroid_assignments']:
            assignments = list(summary['centroid_assignments'].values())
            print(f"  Frame distribution:")
            print(f"    Min: {min(assignments)}")
            print(f"    Max: {max(assignments)}")
            print(f"    Mean: {np.mean(assignments):.1f}")
    else:
        print(f"⚠️ MFCC Training summary not found: {summary_file}")
    
    print("\n" + "=" * 60)
    print("MFCC VERIFICATION COMPLETE")
    print("=" * 60)


def print_usage():
    """Print usage instructions."""
    print("Usage: python analysis_utils.py [feature_type] [data_dir]")
    print("  feature_type: 'lpc' or 'mfcc' (default: lpc)")
    print("  data_dir: path to data directory (default: ../Data)")
    print("\nExamples:")
    print("  python analysis_utils.py lpc")
    print("  python analysis_utils.py mfcc")
    print("  python analysis_utils.py mfcc /path/to/data")


if __name__ == "__main__":
    # Parse command line arguments
    feature_type = "lpc"  # default
    base_dir = "../Data"  # default
    
    if len(sys.argv) > 1:
        if sys.argv[1].lower() in ['lpc', 'mfcc']:
            feature_type = sys.argv[1].lower()
        elif sys.argv[1] in ['-h', '--help', 'help']:
            print_usage()
            sys.exit(0)
        else:
            print(f"Error: Unknown feature type '{sys.argv[1]}'")
            print_usage()
            sys.exit(1)
    
    if len(sys.argv) > 2:
        base_dir = sys.argv[2]
    
    # Run appropriate verification
    print(f"Running {feature_type.upper()} codevector analysis...")
    print(f"Data directory: {base_dir}")
    print()
    
    if feature_type == "lpc":
        verify_codevector_data_lpc(base_dir)
    elif feature_type == "mfcc":
        verify_codevector_data_mfcc(base_dir)
    else:
        print(f"Error: Unsupported feature type '{feature_type}'")
        print_usage()
        sys.exit(1)
