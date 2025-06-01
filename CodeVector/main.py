"""
main.py
Main script for processing audio recordings and creating codevectors.
Each recording is processed separately with its own train/test/HMM split.
"""

import os
import sys
import numpy as np
import random
from pathlib import Path

from codevector_classes import AudioProcessor, DataStorage
from codevector_functions import createCodeVector

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

logger = setup_logger(name="main")



def assign_recordings_to_purposes(data_dir="Data", train_percent=0.5, hmm_percent=0.3, test_percent=0.2):
    """Assign entire recordings to different purposes based on percentages."""
    
    processed_dir = Path(data_dir) / "Processed"
    if not processed_dir.exists():
        print(f"Directory '{processed_dir}' not found.")
        return None
    
    recording_assignments = {
        'train': [],
        'hmm': [],
        'test': []
    }
    
    # Process each word category
    for word_dir in processed_dir.iterdir():
        if word_dir.is_dir():
            word_name = word_dir.name
            print(f"Assigning recordings for word: {word_name}")
            
            # Get all .npy files for this word
            npy_files = list(word_dir.glob("*.npy"))
            npy_files.sort()  # Ensure consistent ordering
            
            total_recordings = len(npy_files)
            if total_recordings == 0:
                print(f"  No recordings found for {word_name}")
                continue
            
            # Calculate split points
            train_end = int(total_recordings * train_percent)
            hmm_end = train_end + int(total_recordings * hmm_percent)
            
            # Assign recordings to purposes
            train_recordings = npy_files[:train_end]
            hmm_recordings = npy_files[train_end:hmm_end]
            test_recordings = npy_files[hmm_end:]
            
            print(f"  Total recordings: {total_recordings}")
            print(f"  Train: {len(train_recordings)} recordings")
            print(f"  HMM: {len(hmm_recordings)} recordings") 
            print(f"  Test: {len(test_recordings)} recordings")
            
            # Store assignments with word category info
            for recording in train_recordings:
                recording_assignments['train'].append((recording, word_name))
            for recording in hmm_recordings:
                recording_assignments['hmm'].append((recording, word_name))
            for recording in test_recordings:
                recording_assignments['test'].append((recording, word_name))
    
    return recording_assignments


def process_recordings_by_purpose(recording_assignments, output_base_dir="Data", print_progress=False):
    """Process recordings according to their assigned purposes."""
    
    if not recording_assignments:
        print("No recording assignments provided!")
        return None, None, None
    
    # Initialize audio processor
    processor = AudioProcessor(
        sample_rate=16000, 
        frame_duration_ms=20,
        overlap_ms=10
    )
    
    print("=" * 60)
    print("Audio Processing Configuration:")
    print(f"  Sample rate: {processor.sample_rate} Hz")
    print(f"  Frame duration: {processor.frame_duration_ms} ms ({processor.frame_size} samples)")
    print(f"  Overlap: {processor.overlap_ms} ms ({processor.overlap_size} samples)")
    print(f"  Hop size: {processor.hop_size} samples")
    print("=" * 60)
    
    storage = DataStorage()
    all_train_frames = []
    all_hmm_frames = []
    all_test_frames = []
    
    # Process each purpose
    for purpose in ['train', 'hmm', 'test']:
        recordings = recording_assignments[purpose]
        print(f"\nProcessing {len(recordings)} recordings for {purpose.upper()} purpose:")
        
        purpose_frames = []
        
        for recording_path, word_name in recordings:
            recording_name = recording_path.stem
            if print_progress:
                print(f"  Processing: {word_name}/{recording_name}")
            
            try:
                # Process entire recording for this purpose
                frames = processor.process_recording(str(recording_path), purpose)
                purpose_frames.extend(frames)
                
                # Create output directory structure based on purpose
                if purpose == 'train':
                    # Training frames go to CodeVector directory (no individual recording separation)
                    # We'll save them all together later
                    pass
                elif purpose == 'hmm':
                    # HMM recordings go to TrainHMM/word/recording/
                    purpose_dir = Path(output_base_dir) / "TrainHMM"
                    word_output_dir = purpose_dir / word_name
                    recording_output_dir = word_output_dir / recording_name
                    recording_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    storage.save_raw_data(frames, recording_output_dir / "hmm_frames.json")
                    storage.save_data_binary(frames, recording_output_dir / "hmm_frames.pkl")
                    
                elif purpose == 'test':
                    # Test recordings go to Test/word/recording/
                    purpose_dir = Path(output_base_dir) / "Test"
                    word_output_dir = purpose_dir / word_name
                    recording_output_dir = word_output_dir / recording_name
                    recording_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    storage.save_raw_data(frames, recording_output_dir / "test_frames.json")
                    storage.save_data_binary(frames, recording_output_dir / "test_frames.pkl")
                
                if print_progress:
                    print(f"    Generated {len(frames)} frames for {purpose}")
                
            except Exception as e:
                print(f"    Error processing {recording_name}: {e}")
                continue
        
        # Store frames by purpose
        if purpose == 'train':
            all_train_frames = purpose_frames
        elif purpose == 'hmm':
            all_hmm_frames = purpose_frames
        elif purpose == 'test':
            all_test_frames = purpose_frames
    
    # Save all training frames together in CodeVector directory
    if all_train_frames:
        codevector_dir = Path(output_base_dir) / "CodeVector"
        codevector_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving all training frames to CodeVector directory...")
        storage.save_raw_data(all_train_frames, codevector_dir / "codevector_frames.json")
        storage.save_data_binary(all_train_frames, codevector_dir / "codevector_frames.pkl")
        print(f"  Saved {len(all_train_frames)} training frames for codevector creation")
        
        # Verify calculations were done properly
        print(f"\nVerifying training frame calculations...")
        frames_with_issues = 0
        for i, frame in enumerate(all_train_frames):
            if (np.allclose(frame.mfcc, 0) and len(frame.raw_samples) > 12):
            # if (np.allclose(frame.lpc_vector, 0) and len(frame.raw_samples) > 12) or \
            #    (np.allclose(frame.lsf_vector, 0) and not np.allclose(frame.lpc_vector, 0)):
                frames_with_issues += 1
        
        if frames_with_issues == 0:
            print(f"  ✓ All frames have proper MFCC calculations")
        else:
            print(f"  ⚠ {frames_with_issues} frames may have calculation issues")
    
    print(f"\nProcessing complete!")
    print(f"Total TRAIN frames: {len(all_train_frames)} (saved in CodeVector/)")
    print(f"Total HMM frames: {len(all_hmm_frames)} (saved in TrainHMM/)")
    print(f"Total TEST frames: {len(all_test_frames)} (saved in Test/)")
    
    return all_train_frames, all_hmm_frames, all_test_frames


EUCLIDIAN = 0
ITAKURA = 1
def create_codevector_from_frames(train_frames, output_dir="Data/CodeVector", centroids_quantity=256, max_iterations = 100, epsilon=0.001, type_distance=EUCLIDIAN):
    """Create codevector from all training frames."""
    
    if not train_frames:
        print("No training frames available!")
        return None, None
    
    # Create CodeVector output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print(f"Creating codevector with {centroids_quantity} centroids")
    print(f"Epsilon: {epsilon}")
    print(f"Training frames from all recordings: {len(train_frames)}")
    print("=" * 60)
    
    # Create codevector (now saves updated frames automatically)
    centroids, generations = createCodeVector(
        raw_data_vocabulary=train_frames,
        centroids_quantity=centroids_quantity,
        max_iterations=max_iterations,
        epsilon=epsilon,
        save_updates=True,  # Enable saving updated frames
        output_dir=output_dir  # Provide output directory
        # type_distance=type_distance
    )
    
    # Save results
    storage = DataStorage()
    storage.save_centroids(centroids, os.path.join(output_dir, "codevector.json"))
    storage.save_generations(generations, os.path.join(output_dir, "generations.json"))
    
    # Also save as binary
    storage.save_data_binary(centroids, os.path.join(output_dir, "codevector.pkl"))
    storage.save_data_binary(generations, os.path.join(output_dir, "generations.pkl"))
    
    print(f"\nFinal codevector has {len(centroids)} centroids")
    print(f"Created {len(generations)} generations")
    print(f"Codevector saved to: {output_dir}")
    
    return centroids, generations


def load_training_frames_from_file(base_dir="../Data"):
    """Load training frames from saved files for codevector creation."""
    
    codevector_dir = Path(base_dir) / "CodeVector"
    frames_file = codevector_dir / "codevector_frames.pkl"
    
    if not frames_file.exists():
        print(f"Training frames file not found: {frames_file}")
        print("Please run steps 1 and 2 first to generate training data.")
        return None
    
    storage = DataStorage()
    try:
        train_frames = storage.load_data_binary(str(frames_file))
        print(f"Loaded {len(train_frames)} training frames from file")
        return train_frames
    except Exception as e:
        print(f"Error loading training frames: {e}")
        return None

def demonstrate_loading(base_dir="Data"):
    """Demonstrate loading saved data from different recordings."""
    
    print("\n" + "=" * 60)
    print("Demonstrating data loading...")
    print("=" * 60)
    
    storage = DataStorage()
    
    # Load codevector
    codevector_dir = os.path.join(base_dir, "CodeVector")
    if os.path.exists(os.path.join(codevector_dir, "codevector.json")):
        print("\nLoading codevector:")
        centroids = storage.load_centroids(os.path.join(codevector_dir, "codevector.json"))
        generations = storage.load_generations(os.path.join(codevector_dir, "generations.json"))
        print(f"  Loaded codevector with {len(centroids)} centroids")
        
        # Load updated training frames if available
        updated_frames_file = os.path.join(codevector_dir, "codevector_frames_updated.pkl")
        if os.path.exists(updated_frames_file):
            print("\nLoading updated training frames:")
            updated_frames = storage.load_data_binary(updated_frames_file)
            print(f"  Loaded {len(updated_frames)} updated training frames")
            
            # Show some statistics
            generations_used = set(frame.generation for frame in updated_frames)
            centroids_used = set(frame.parent_centroid_id for frame in updated_frames)
            print(f"  Generations used: {sorted(generations_used)}")
            print(f"  Number of centroids assigned: {len(centroids_used)}")
            
            # Show sample frame info
            if updated_frames:
                sample_frame = updated_frames[0]
                print(f"  Sample frame info:")
                print(f"    Recording: {sample_frame.recording}")
                print(f"    Frame number: {sample_frame.frame_number}")
                print(f"    Generation: {sample_frame.generation}")
                print(f"    Parent centroid ID: {sample_frame.parent_centroid_id}")
                print(f"    MFCC calculated: {not np.allclose(sample_frame.mfcc, 0)}")
                # print(f"    LSF calculated: {not np.allclose(sample_frame.lsf_vector, 0)}")
                # print(f"    LPC calculated: {not np.allclose(sample_frame.lpc_vector, 0)}")
        
        # Load and show training summary
        summary_file = os.path.join(codevector_dir, "training_summary.json")
        if os.path.exists(summary_file):
            print("\nTraining Summary:")
            with open(summary_file, 'r') as f:
                import json
                summary = json.load(f)
            print(f"  Total frames processed: {summary['total_frames']}")
            print(f"  Maximum generation: {summary['max_generation']}")
            print(f"  Centroids with assigned frames: {len(summary['centroid_assignments'])}")
            
            # Show centroid usage distribution
            assignments = list(summary['centroid_assignments'].values())
            if assignments:
                print(f"  Frame distribution per centroid:")
                print(f"    Min frames per centroid: {min(assignments)}")
                print(f"    Max frames per centroid: {max(assignments)}")
                print(f"    Average frames per centroid: {np.mean(assignments):.1f}")
    else:
        print(f"\nCodevector not found in {codevector_dir}")
        print("Run codevector creation step first.")
    
    # Load sample recording data with random selection
    print("\nLoading sample recording data:")
    base_path = Path(base_dir)
    
    # Collect all available recordings from both TrainHMM and Test directories
    all_recordings = []
    for purpose_dir in ["TrainHMM", "Test"]:
        purpose_path = base_path / purpose_dir
        if purpose_path.exists():
            for word_dir in purpose_path.iterdir():
                if word_dir.is_dir():
                    word_name = word_dir.name
                    for recording_dir in word_dir.iterdir():
                        if recording_dir.is_dir():
                            # Look for frame files
                            frame_files = list(recording_dir.glob("*_frames.json"))
                            if frame_files:
                                all_recordings.append({
                                    'purpose': purpose_dir,
                                    'word': word_name,
                                    'recording_dir': recording_dir,
                                    'frame_file': frame_files[0]
                                })
    
    if all_recordings:
        # Randomly select one recording
        selected_recording = random.choice(all_recordings)
        
        print(f"  Randomly selected recording:")
        print(f"    Purpose: {selected_recording['purpose']}")
        print(f"    Word: {selected_recording['word']}")
        
        # Load the frames
        # frames = storage.load_raw_data_(str(selected_recording['frame_file']))
        frames = storage.load_raw_data_mfcc(str(selected_recording['frame_file']))
        print(f"    Frames: {len(frames)}")
        
        # Show some sample data
        if frames:
            # Show a few more random frame examples if available
            if len(frames) > 1:
                num_samples = min(3, len(frames))
                random_indices = random.sample(range(len(frames)), num_samples)
                print(f"    Additional random frame samples:")
                for i, idx in enumerate(random_indices):
                    frame = frames[idx]
                    print(f"      Frame {idx}: Recording {frame.recording}, Frame #{frame.frame_number}")
    else:
        print("  No recording data found to demonstrate loading")
        print("  Run processing steps first to generate data.")
    
    print(f"\nData loading demonstration complete!")


def print_usage():
    """Print usage information."""
    print("Usage:")
    print("  python main.py          - Show this help message")
    print("  python main.py a        - Run all steps (1, 2, CodeVector, Loading)")
    print("  python main.py 2        - Run only steps 1 and 2 (assignment and processing)")
    print("  python main.py code     - Run only CodeVector step")
    print("  python main.py load     - Run only Loading step")


def main():
    """Main function to run the complete workflow based on command line arguments."""
    
    print("Codevector Creation System - Recording-Based Purpose Assignment")
    print("=" * 60)
    
    # Parse command line arguments
    if len(sys.argv) > 2:
        print("Error: Too many arguments provided.")
        print_usage()
        return
    
    # Determine which steps to run
    if len(sys.argv) == 1:
        # No arguments - show usage instructions
        print_usage()
        return
    elif sys.argv[1].lower() == 'a':
        # Run all steps
        run_steps = ['assignment', 'processing', 'codevector', 'loading']
        print("Running all steps (1, 2, CodeVector, Loading)")
    elif sys.argv[1] == '2':
        # Run only steps 1 and 2
        run_steps = ['assignment', 'processing']
        print("Running steps 1 and 2 only (assignment and processing)")
    elif sys.argv[1].lower() == 'code':
        # Run only codevector step
        run_steps = ['codevector']
        print("Running CodeVector step only")
    elif sys.argv[1].lower() == 'load':
        # Run only loading step
        run_steps = ['loading']
        print("Running Loading step only")
    else:
        print(f"Error: Unknown argument '{sys.argv[1]}'")
        print_usage()
        return
    
    # Configuration
    data_dir = "../Data"
    output_base_dir = "../Data"
    
    # Variables to store data between steps
    recording_assignments = None
    all_train_frames = None
    all_hmm_frames = None
    all_test_frames = None
    
    # Step 1: Assignment
    if 'assignment' in run_steps:
        print("\nStep 1: Assigning recordings to purposes...")
        print("Each entire recording will be used for ONE purpose:")
        print("  - 40% of recordings → Codevector training")
        print("  - 30% of recordings → HMM training") 
        print("  - 30% of recordings → Testing")
        
        # Check if data directory exists
        if not os.path.exists(os.path.join(data_dir, "Processed")):
            print(f"Data directory '{data_dir}/Processed' not found.")
            print("Expected structure:")
            print("  Data/Processed/word1/recording1.npy")
            print("  Data/Processed/word1/recording2.npy")
            print("  Data/Processed/word2/recording1.npy")
            print("  ...")
            print("Exiting...")
            return
        
        recording_assignments = assign_recordings_to_purposes(data_dir,
                                                              train_percent = 0.3,
                                                              hmm_percent=0.5, 
                                                              test_percent=0.2)
        
        if not recording_assignments:
            print("Failed to assign recordings. Exiting...")
            return
    
    # Step 2: Processing
    if 'processing' in run_steps:
        print("\nStep 2: Processing recordings according to their assigned purposes...")
        
        if recording_assignments is None:
            print("Error: No recording assignments available.")
            print("Please run step 1 first or run with no arguments to run all steps.")
            return
        
        all_train_frames, all_hmm_frames, all_test_frames = process_recordings_by_purpose(
            recording_assignments, 
            output_base_dir
        )
        
        if all_train_frames is None:
            print("Failed to process recordings. Exiting...")
            return
    
    # Step CodeVector: Create codevector
    if 'codevector' in run_steps:
        print("\nStep 3: Creating codevector from TRAINING recordings only...")
        
        # If we don't have training frames from previous steps, try to load them
        if all_train_frames is None:
            all_train_frames = load_training_frames_from_file(output_base_dir)
            
        if all_train_frames is None:
            print("No training frames available for codevector creation.")
            print("Please run steps 1 and 2 first.")
            return
        
        codevector_output_dir = os.path.join(output_base_dir, "CodeVector")
        centroids, generations = create_codevector_from_frames(
            all_train_frames, 
            codevector_output_dir,
            centroids_quantity=256,
            max_iterations=100,
            # epsilon=0.001,
            type_distance=ITAKURA
        )
        
        if centroids is None:
            print("Failed to create codevector. Exiting...")
            return
    
    # Step Loading: Demonstrate loading
    if 'loading' in run_steps:
        print("\nStep 4: Demonstrating data loading...")
        demonstrate_loading(output_base_dir)
    
    print("\n" + "=" * 60)
    print("Execution complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
