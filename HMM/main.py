#!/usr/env/bin python3

import sys
import os
import random
from pathlib import Path
from collections import defaultdict
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hmm_training import training_with_save 
from hmm_classes import DataStorageHMM
from hmm_testing import test_hmm, create_confusion_matrix 
from CodeVector.codevector_classes import RawDataMFCC, DataStorage

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


def load_all_recordings_by_word(base_dir="../Data", purpose="TrainHMM", print_messages=True, print_summary=True) -> dict[str, list[list[RawDataMFCC]]]:
    """
    Args:
        purpose: Purpose directory to load from ("TrainHMM" or "Test")
    Returns:
        dict: Dictionary where keys are word names and values are lists of recordings
              Format: {"word1": [recording1_frames, recording2_frames, ...], "word2": [...], ...}
              Where each recording_frames is a list[RawDataMFCC]
              So the structure is: dict[str, list[list[RawDataMFCC]]]
    """
    storage = DataStorage()
    base_path = Path(base_dir)
    
    # Use defaultdict to automatically create empty lists for new words
    all_words = defaultdict(list)
    recording_counts = defaultdict(int)  # Track number of recordings per word
    
    purpose_path = base_path / purpose
    if not purpose_path.exists():
        print(f"Warning: Directory {purpose_path} does not exist")
        return dict(all_words)
    
    if print_messages:
        print(f"Loading recordings from {purpose_path}")
    
    for word_dir in purpose_path.iterdir():  # Eg. finish/
        if word_dir.is_dir():
            word_name = word_dir.name
            if print_messages:
                print(f"  Processing word: {word_name}")
            
            for recording_dir in word_dir.iterdir():  # Eg. finish-04/
                if recording_dir.is_dir():
                    # Look for frame files
                    frame_files = list(recording_dir.glob("*_frames.json"))
                    if frame_files:
                        # Load the frames for this recording
                        frames = storage.load_raw_data_mfcc(str(frame_files[0]), print_messages=print_messages)
                        if frames:  # Only add if frames were successfully loaded
                            # Add the entire recording as a single list instead of extending
                            all_words[word_name].append(frames)
                            recording_counts[word_name] += 1
                            if print_messages:
                                print(f"    Added recording with {len(frames)} frames from {recording_dir.name}")
    
    # Convert defaultdict to regular dict for cleaner output
    result = dict(all_words)
    
    # Print summary
    if print_summary:
        print(f"\nSummary:")
        print(f"  Total words: {len(result)}")
        for word, recordings in result.items():
            total_frames = sum(len(recording) for recording in recordings)
            print(f"    {word}: {len(recordings)} recordings with {total_frames} total frames")
    
    return result


def load_mfcc_centroids(base_dir="../Data", print_messages=True):
    """
    Returns:
        list: [CentroidDataMFCC, ..., CentroidDataMFCC]
    """

    centroids = []
    storage = DataStorage()
    # Load codevector
    codevector_dir = os.path.join(base_dir, "CodeVector")
    if os.path.exists(os.path.join(codevector_dir, "codevector.json")):
        if print_messages:
            print("\nLoading codevector:")
        centroids = storage.load_centroids(os.path.join(codevector_dir, "codevector.json"))
        if print_messages:
            print(f"  Loaded codevector with {len(centroids)} centroids")
            print(f"  Example random centroid:")
            random_centroid = random.choice(centroids)
            print(f"   id: {random_centroid.id}")
            print(f"   Power: {random_centroid.mfcc[0]:.3f}")
            for i in range(1, random_centroid.mfcc.shape[0]):
                print(f"   {random_centroid.mfcc[i]:.3f}", end=" ")

            print(f"\n")

    return centroids


def train_hmm(show_progress=True, max_iterations=100, load_initial_params=False):
    """Train HMM using Baum-Welch algorithm for all words"""
    print("Starting HMM training for all words...")
    try:
        # Load code vector (centroids)
        centroids = load_mfcc_centroids(print_messages=False)
        print(f"Loaded {len(centroids)} centroids")
        
        # Load training recordings
        recordings_by_word = load_all_recordings_by_word(purpose="TrainHMM", print_messages=False)
        print(f"Loaded recordings for {len(recordings_by_word)} words")
        
        # Train HMM for each word
        trained_hmms = []
        for word_name, word_recordings in recordings_by_word.items():
            print(f"\nTraining HMM for word: '{word_name}' with {len(word_recordings)} recordings")
            
            # Perform training and save
            hmm_model = training_with_save(word_recordings, centroids, word_name, max_iterations = max_iterations, show_progress=show_progress, load_initial_params=load_initial_params)
            trained_hmms.append(hmm_model)
            
            print(f"Model saved for word: '{hmm_model.word}'")
        
        print(f"\nHMM training completed successfully!")
        print(f"Total models trained: {len(trained_hmms)}")
        print(f"Words trained: {[hmm.word for hmm in trained_hmms]}")
        
        return trained_hmms
        
    except Exception as e:
        logger.error(f"Error during HMM training: {e}")
        return None


def test(show_progress=False):
    """Test trained HMM models on test data."""
    print("Loading trained HMM models...")
    all_hmm = DataStorageHMM.load_all_hmms()
    
    if not all_hmm:
        print("No trained HMM models found. Please train models first.")
        return
    
    print(f"Loaded {len(all_hmm)} HMM models for words: {[hmm.word for hmm in all_hmm]}")
    
    # Load test recordings
    test_recordings_dict = load_all_recordings_by_word(purpose="Test", print_messages=False)
    print(f"Loaded test recordings for {len(test_recordings_dict)} words")
    
    # Filter test recordings to only include words we have trained models for
    trained_words = {hmm.word for hmm in all_hmm}
    filtered_test_recordings = {word: recordings for word, recordings in test_recordings_dict.items() 
                               if word in trained_words}

    if not filtered_test_recordings:
        print("No test recordings found for trained words.")
        return
    
    print(f"Testing on {len(filtered_test_recordings)} words: {list(filtered_test_recordings.keys())}")
    
    # Perform testing
    true_labels, predicted_labels = test_hmm(all_hmm, filtered_test_recordings, show_progress=False)
    
    # Create and save confusion matrix
    create_confusion_matrix(true_labels, predicted_labels)
   
def demo_usage():
    """Demonstrate how to use the loaded data."""
    print("="*50)
    print("DEMO: Loading and using the organized recordings")
    print("="*50)
    
    # Load all recordings organized by word
    all_words = load_all_recordings_by_word(print_messages=False)
    
    if not all_words:
        print("No recordings found!")
        return
    
    # Show available words
    print(f"\nAvailable words: {list(all_words.keys())}")
    
    # Show 3 random words with random frames
    available_words = list(all_words.keys())
    if len(available_words) >= 3:
        random_words = random.sample(available_words, 3)
    else:
        random_words = available_words
    
    print(f"\nRandom word examples:")
    for word_name in random_words:
        recordings = all_words[word_name]  # This is list[list[RawDataMFCC]]
        if recordings:
            # Step 1: Randomly choose one recording from the word's recordings
            random_recording = random.choice(recordings)  # This is list[RawDataMFCC]
            
            # Step 2: From that recording, randomly choose one frame
            random_frame = random.choice(random_recording)  # This is RawDataMFCC

            
            print(f"  Word '{word_name}' (total recordings: {len(recordings)}):")
            print(f"    Random frame details:")
            print(f"      Recording: {random_frame.recording}")
            print(f"      Frame number: {random_frame.frame_number}")
            print(f"      MFCC shape: {random_frame.mfcc.shape}")
            print(f"      Raw samples shape: {random_frame.raw_samples.shape}")
            print(f"      Sample rate: {random_frame.sample_rate}")
            print()

    print("="*50)
    print("DEMO: Loading Centroids")
    print("="*50)
    load_mfcc_centroids()


def show_menu():
    """Display simple menu with just load option."""
    print("="*50)
    print("AUDIO RECORDINGS LOADER")
    print("="*50)
    print("Options:")
    print("  python main.py      -> Show this menu")
    print("  python main.py load -> Run demo")
    print("  python main.py train -> Run train")
    print("  python main.py test -> Run test")
    print("="*50)

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1].lower() == 'load':
            # Run demo if 'load' parameter is provided
            demo_usage()
        elif sys.argv[1].lower() == 'train':
            # Run HMM training if 'train' parameter is provided
            train_hmm(show_progress=True, max_iterations=2)
        elif sys.argv[1].lower() == 'test':
            # Run HMM testing if 'test' parameter is provided
            test()
        else:
            print("Unknown parameter. Use 'load' for demo, 'train' for HMM training, or 'test' for HMM testing.")
    else:
        # Show interactive menu by default
        show_menu()
