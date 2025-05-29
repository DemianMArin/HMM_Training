"""
codevector_functions.py
Functions and utilities for the codevector creation system.
"""

import numpy as np
import json
import pickle
import os
from pathlib import Path
from typing import List, Tuple
import librosa
import random
from codevector_classes import RawData, CentroidData, RawDataMFCC, CentroidDataMFCC, DataStorage

import sys
import logging


def setup_logger(name=__name__, level=logging.DEBUG):
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding multiple handlers if logger already exists
    if logger.handlers:
        return logger
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create formatter with file, method, and line information
    formatter = logging.Formatter(
        fmt=' %(levelname)-8s | %(filename)s:%(lineno)d | %(funcName)s() | %(message)s',
    )
    
    # Add formatter to handler
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logger(name="functions")

def print_array(name: str, vector: np.ndarray):
    flat_ = vector.flatten()
    print(f"{name}", end=" ")
    for i in flat_:
        print(f"{i:.4f}",end=" ")
    print(f"\n")

def itakura_saito_distance(auto_coeff_raw: np.ndarray, lpc_centroid: np.ndarray) -> float:
    """
    Compute Itakura-Saito distance with short correlation:
    dIS = r[0]*ra[0] + 2*sum(r[n]*ra[n]) for n=1 to P
    """

    if(auto_coeff_raw.shape[0] != lpc_centroid.shape[0]):
        logger.error(f"Size differ")

    ra = librosa.autocorrelate(lpc_centroid,max_size=len(auto_coeff_raw))
    
    # Apply the formula
    distance = auto_coeff_raw[0] * ra[0]
    for n in range(1, len(auto_coeff_raw)):
        distance += 2 * auto_coeff_raw[n] * ra[n]


    # if random.randint(1,5380) == 1:
    if False:
        print_array("r[:] ", auto_coeff_raw)
        print_array("lpc coeff ", lpc_centroid)
        print_array("ra[:] ", ra)
        print(f"dist: {distance}, |dist|: {abs(distance)}")

    return abs(distance)


def euclidian_distance(lsf_coeff_raw: np.ndarray, lsf_centroid: np.ndarray) -> float:
    if len(lsf_coeff_raw) !=  len(lsf_centroid):
        raise ValueError("Vectors must be of size 13.")

    distance = np.linalg.norm(np.array(lsf_coeff_raw) - np.array(lsf_centroid))
    return distance


def new_epsilon_centroids(centroids: List[CentroidData], alpha1: float = 1.01, alpha2: float = 0.99) -> List[CentroidData]:
    """Create new centroids by splitting existing ones with epsilon perturbation."""
    n_centroids = len(centroids)
    
    # Check if number of centroids is power of 2
    if n_centroids & (n_centroids - 1) != 0 or n_centroids == 0:
        print(f"Warning: Number of centroids ({n_centroids}) is not a power of 2")
        return centroids
    
    new_centroids = []
    
    for i, centroid in enumerate(centroids):
        # Create two new centroids from each existing one
        # Even index: multiply by alpha1
        new_centroid_even = CentroidData(
            lsf=centroid.lsf * alpha1,
            id=2 * i
        )
        
        # Odd index: multiply by alpha2
        new_centroid_odd = CentroidData(
            lsf=centroid.lsf * alpha2,
            id=2 * i + 1
        )
        
        new_centroids.append(new_centroid_even)
        new_centroids.append(new_centroid_odd)
    
    return new_centroids


def new_adjust_centroids(raw_data_vocabulary: List[RawData]) -> List[CentroidData]:
    """Adjust centroids based on assigned raw data frames."""
    if not raw_data_vocabulary:
        return []
    
    # Get generation (should be same for all)
    generation = max(frame.generation for frame in raw_data_vocabulary)
    number_of_centroids = 2 ** generation
    
    centroids = []
    
    for centroid_id in range(number_of_centroids):
        # Find all frames assigned to this centroid
        assigned_frames = [frame for frame in raw_data_vocabulary 
                          if frame.parent_centroid_id == centroid_id]
        
        if assigned_frames:
            # Calculate mean LSF for assigned frames
            mean_lsf = np.mean([frame.lsf_vector for frame in assigned_frames], axis=0)
            centroid = CentroidData(lsf=mean_lsf, id=centroid_id)
        else:
            centroid = CentroidData(lsf=np.zeros(12), id=centroid_id)
        
        centroids.append(centroid)
    
    return centroids


def verify_frame_calculations(raw_data_vocabulary: List[RawData]) -> bool:
    """Verify that all frames have proper calculations done."""
    print("Verifying frame calculations...")
    
    issues_found = 0
    total_frames = len(raw_data_vocabulary)
    
    for i, frame in enumerate(raw_data_vocabulary):
        # Check if autocorrelation is calculated (should not be all zeros unless frame was empty)
        if np.allclose(frame.autocorrelation_vector, 0) and len(frame.raw_samples) > 0:
            print(f"  Warning: Frame {i} has zero autocorrelation but non-empty raw samples")
            issues_found += 1
        
        # Check if LPC is calculated (should not be all zeros for valid frames)
        if np.allclose(frame.lpc_vector, 0) and len(frame.raw_samples) > 12:
            print(f"  Warning: Frame {i} has zero LPC but sufficient samples")
            issues_found += 1
        
        # Check if LSF is calculated (should not be all zeros if LPC exists)
        if np.allclose(frame.lsf_vector, 0) and not np.allclose(frame.lpc_vector, 0):
            print(f"  Warning: Frame {i} has zero LSF but non-zero LPC")
            issues_found += 1
    
    if issues_found == 0:
        print(f"  ✓ All {total_frames} frames have proper calculations")
        return True
    else:
        print(f"  ✗ Found {issues_found} issues in {total_frames} frames")
        return False


def save_updated_training_frames(raw_data_vocabulary: List[RawData], output_dir: str):
    """Save the updated training frames with generation and parent_centroid_id info."""
    print(f"Saving updated training frames to {output_dir}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    storage = DataStorage()
    
    # Save updated frames
    updated_frames_path = os.path.join(output_dir, "codevector_frames_updated.json")
    updated_frames_pkl_path = os.path.join(output_dir, "codevector_frames_updated.pkl")
    
    storage.save_raw_data(raw_data_vocabulary, updated_frames_path)
    storage.save_data_binary(raw_data_vocabulary, updated_frames_pkl_path)
    
    # Create a summary for verification
    summary = {
        'total_frames': len(raw_data_vocabulary),
        'max_generation': max(frame.generation for frame in raw_data_vocabulary) if raw_data_vocabulary else 0,
        'centroid_assignments': {}
    }
    
    # Count frames per centroid
    for frame in raw_data_vocabulary:
        centroid_id = frame.parent_centroid_id
        if centroid_id not in summary['centroid_assignments']:
            summary['centroid_assignments'][centroid_id] = 0
        summary['centroid_assignments'][centroid_id] += 1
    
    summary_path = os.path.join(output_dir, "training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  Saved updated frames: {updated_frames_path}")
    print(f"  Saved training summary: {summary_path}")
    print(f"  Total frames: {summary['total_frames']}")
    print(f"  Max generation: {summary['max_generation']}")
    print(f"  Centroids used: {len(summary['centroid_assignments'])}")

EUCLIDIAN = 0
ITAKURA = 1
def createCodeVector(raw_data_vocabulary: List[RawData], centroids_quantity: int = 256, max_iterations = 100, epsilon: float = 0.001, save_updates: bool = True, output_dir: str = None, type_distance=EUCLIDIAN) -> Tuple[List[CentroidData], List[List[CentroidData]]]:
    """Create codevector using LBG algorithm."""
    if not raw_data_vocabulary:
        raise ValueError("No raw data provided")
    
    print(f"Creating codevector with {centroids_quantity} centroids...")
    print(f"Using {len(raw_data_vocabulary)} frames for training")
    
    # Verify that all frames have proper calculations
    print("\n" + "="*50)
    if not verify_frame_calculations(raw_data_vocabulary):
        print("Warning: Some frames may have calculation issues")
    print("="*50)
    
    # Calculate initial centroid C0
    all_lsf = np.array([frame.lsf_vector for frame in raw_data_vocabulary])
    c0_lsf = np.mean(all_lsf, axis=0)
   
    # Create initial centroid
    centroids = [CentroidData(lsf=c0_lsf, id=0)]
    
    # Calculate number of generations
    generations_num = int(np.log2(centroids_quantity))
    generations = [centroids]
    
    # Epsilon split for first generation
    centroids = new_epsilon_centroids(centroids)
    
    # LBG algorithm iterations
    for generation in range(1, generations_num + 1):
        print(f"\nGeneration {generation}: Creating {len(centroids)} centroids")
        
        global_dist_prev = 0
        diff = epsilon + 100  # Ensure at least one iteration
        
        # Set generation for all frames
        for frame in raw_data_vocabulary:
            frame.generation = generation
        
        iteration_count = 0
        max_iterations = max_iterations
        
        while diff > epsilon and iteration_count < max_iterations:
            iteration_count += 1
            global_dist = 0

            # Assign each frame to nearest centroid
            for frame in raw_data_vocabulary:
                min_dist = float('inf')
                best_centroid_id = 0
               
                for centroid in centroids:
                    if type_distance == EUCLIDIAN:
                        dist = euclidian_distance(frame.lsf_vector, centroid.lsf)
                    else:
                        dist = itakura_saito_distance(frame.autocorrelation_vector, centroid.lpc)

                    if dist < min_dist:
                        min_dist = dist
                        best_centroid_id = centroid.id

                frame.parent_centroid_id = best_centroid_id
                global_dist += min_dist

            # Adjust centroids based on assignments
            centroids = new_adjust_centroids(raw_data_vocabulary)
            
            # Calculate difference
            diff = abs(global_dist_prev - global_dist)
            global_dist_prev = global_dist
            
            if (iteration_count % 1 == 0 and type_distance == EUCLIDIAN):
                print(f"  Iteration {iteration_count}: dist={global_dist:.6f}, diff={diff:.6f}")
            if (iteration_count % 10 == 0 and type_distance == ITAKURA):
                print(f"  Iteration {iteration_count}: dist={global_dist:.6f}, diff={diff:.6f}")

        
        print(f"  Converged after {iteration_count} iterations (diff={diff:.6f})")
        generations.append(centroids)
        
        # Split centroids for next generation (if not last)
        if generation < generations_num:
            centroids = new_epsilon_centroids(centroids)
    
    print(f"\nCodevector creation complete!")
    
    # Save updated training frames if requested
    if save_updates and output_dir:
        print("\n" + "="*50)
        save_updated_training_frames(raw_data_vocabulary, output_dir)
        print("="*50)
    
    return centroids, generations





def verify_frame_calculations(raw_data_vocabulary: List[RawDataMFCC]) -> bool:
    """Verify that all frames have proper calculations done."""
    print("Verifying frame calculations...")
    
    issues_found = 0
    total_frames = len(raw_data_vocabulary)
    
    for i, frame in enumerate(raw_data_vocabulary):
        # Check if MFCC is calculated (should not be all zeros if MFCC exists)
        if np.allclose(frame.mfcc, 0) and not np.allclose(frame.mfcc, 0):
            print(f"  Warning: Frame {i} has zero MFCC")
            issues_found += 1
    
    if issues_found == 0:
        print(f"  ✓ All {total_frames} frames have proper calculations")
        return True
    else:
        print(f"  ✗ Found {issues_found} issues in {total_frames} frames")
        return False


def save_updated_training_frames(raw_data_vocabulary: List[RawDataMFCC], output_dir: str):
    """Save the updated training frames with generation and parent_centroid_id info."""
    print(f"Saving updated training frames to {output_dir}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    storage = DataStorage()
    
    # Save updated frames
    updated_frames_path = os.path.join(output_dir, "codevector_frames_updated.json")
    updated_frames_pkl_path = os.path.join(output_dir, "codevector_frames_updated.pkl")
    
    storage.save_raw_data(raw_data_vocabulary, updated_frames_path)
    storage.save_data_binary(raw_data_vocabulary, updated_frames_pkl_path)
    
    # Create a summary for verification
    summary = {
        'total_frames': len(raw_data_vocabulary),
        'max_generation': max(frame.generation for frame in raw_data_vocabulary) if raw_data_vocabulary else 0,
        'centroid_assignments': {}
    }
    
    # Count frames per centroid
    for frame in raw_data_vocabulary:
        centroid_id = frame.parent_centroid_id
        if centroid_id not in summary['centroid_assignments']:
            summary['centroid_assignments'][centroid_id] = 0
        summary['centroid_assignments'][centroid_id] += 1
    
    summary_path = os.path.join(output_dir, "training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  Saved updated frames: {updated_frames_path}")
    print(f"  Saved training summary: {summary_path}")
    print(f"  Total frames: {summary['total_frames']}")
    print(f"  Max generation: {summary['max_generation']}")
    print(f"  Centroids used: {len(summary['centroid_assignments'])}")


def new_epsilon_centroids(centroids: List[CentroidDataMFCC], alpha1: float = 1.001, alpha2: float = 0.999) -> List[CentroidDataMFCC]:
    """Create new centroids by splitting existing ones with epsilon perturbation."""
    n_centroids = len(centroids)
    
    # Check if number of centroids is power of 2
    if n_centroids & (n_centroids - 1) != 0 or n_centroids == 0:
        print(f"Warning: Number of centroids ({n_centroids}) is not a power of 2")
        return centroids
    
    new_centroids = []
    
    for i, centroid in enumerate(centroids):
        # Create two new centroids from each existing one
        # Even index: multiply by alpha1
        new_centroid_even = CentroidDataMFCC(
            mfcc=centroid.mfcc * alpha1,
            id=2 * i
        )
        
        # Odd index: multiply by alpha2
        new_centroid_odd = CentroidDataMFCC(
            mfcc=centroid.mfcc * alpha2,
            id=2 * i + 1
        )
        
        new_centroids.append(new_centroid_even)
        new_centroids.append(new_centroid_odd)
    
    return new_centroids


def new_adjust_centroids(raw_data_vocabulary: List[RawDataMFCC]) -> List[CentroidDataMFCC]:
    """Adjust centroids based on assigned raw data frames."""
    if not raw_data_vocabulary:
        return []
    
    # Get generation (should be same for all)
    generation = max(frame.generation for frame in raw_data_vocabulary)
    number_of_centroids = 2 ** generation
    
    centroids = []
    
    for centroid_id in range(number_of_centroids):
        # Find all frames assigned to this centroid
        assigned_frames = [frame for frame in raw_data_vocabulary 
                          if frame.parent_centroid_id == centroid_id]
        
        if assigned_frames:
            # Calculate mean LSF for assigned frames
            mean_mfcc= np.mean([frame.mfcc for frame in assigned_frames], axis=0)
            centroid = CentroidDataMFCC(mfcc=mean_mfcc, id=centroid_id)
        else:
            centroid = CentroidDataMFCC(mfcc=np.zeros(13), id=centroid_id)
        
        centroids.append(centroid)
    
    return centroids


def createCodeVector(raw_data_vocabulary: List[RawDataMFCC], centroids_quantity: int = 256, max_iterations = 100, epsilon: float = 0.001, save_updates: bool = True, 
                         output_dir: str = None) -> Tuple[List[CentroidDataMFCC], List[List[CentroidDataMFCC]]]:
    """Create codevector using LBG algorithm."""
    if not raw_data_vocabulary:
        raise ValueError("No raw data provided")
    
    print(f"Creating codevector with {centroids_quantity} centroids...")
    print(f"Using {len(raw_data_vocabulary)} frames for training")
    
    # Verify that all frames have proper calculations
    print("\n" + "="*50)
    if not verify_frame_calculations(raw_data_vocabulary):
        print("Warning: Some frames may have calculation issues")
    print("="*50)
    
    # Calculate initial centroid C0
    all_mfcc = np.array([frame.mfcc for frame in raw_data_vocabulary])
    c0_mfcc = np.mean(all_mfcc , axis=0)
   
    # Create initial centroid
    centroids = [CentroidDataMFCC(mfcc=c0_mfcc, id=0)]
    
    # Calculate number of generations
    generations_num = int(np.log2(centroids_quantity))
    generations = [centroids]
    
    # Epsilon split for first generation
    centroids = new_epsilon_centroids(centroids)
    
    # LBG algorithm iterations
    for generation in range(1, generations_num + 1):
        print(f"\nGeneration {generation}: Creating {len(centroids)} centroids")
        
        global_dist_prev = 0
        diff = epsilon + 100  # Ensure at least one iteration
        
        # Set generation for all frames
        for frame in raw_data_vocabulary:
            frame.generation = generation
        
        iteration_count = 0
        max_iterations = max_iterations
        
        while diff > epsilon and iteration_count < max_iterations:
            iteration_count += 1
            global_dist = 0

            # Assign each frame to nearest centroid
            for frame in raw_data_vocabulary:
                min_dist = float('inf')
                best_centroid_id = 0
               
                for centroid in centroids:
                    # Index 0 is power, Calculating distance with only spectral shape
                    dist = euclidian_distance(frame.mfcc[1:], centroid.mfcc[1:])

                    if dist < min_dist:
                        min_dist = dist
                        best_centroid_id = centroid.id

                frame.parent_centroid_id = best_centroid_id
                global_dist += min_dist

            # Adjust centroids based on assignments
            centroids = new_adjust_centroids(raw_data_vocabulary)
            
            # Calculate difference
            diff = abs(global_dist_prev - global_dist)
            global_dist_prev = global_dist
            
            if (iteration_count % 10 == 0):
                print(f"  Iteration {iteration_count}: dist={global_dist:.6f}, diff={diff:.6f}")

        
        print(f"  Converged after {iteration_count} iterations (diff={diff:.6f})")
        generations.append(centroids)
        
        # Split centroids for next generation (if not last)
        if generation < generations_num:
            centroids = new_epsilon_centroids(centroids)
    
    print(f"\nCodevector creation complete!")
    
    # Save updated training frames if requested
    if save_updates and output_dir:
        print("\n" + "="*50)
        save_updated_training_frames(raw_data_vocabulary, output_dir)
        print("="*50)
    
    return centroids, generations
