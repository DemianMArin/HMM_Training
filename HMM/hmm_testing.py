
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Tuple
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from CodeVector.codevector_classes import DataStorage, RawDataMFCC
from hmm_training import get_observations, calculate_log_alpha, safe_log, log_sum_exp
from hmm_classes import HMMTrained




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

logger = setup_logger(name="testing")

def calculate_log_likelihood(recording_observations: np.ndarray, hmm: HMMTrained) -> float:
    """
    Calculate log-likelihood of a single recording given an HMM model using forward algorithm.
    Uses log-space calculations to prevent underflow.
    
    Args:
        recording_observations: Single recording observation sequence (np.ndarray)
                               Shape: (T,) where T is sequence length
                               Example: [45, 12, 203, 67, ...] representing centroid IDs
        hmm: Trained HMM model containing linear-space matrices
    
    Returns:
        float: log P(O|λ) - log-likelihood for this recording (for direct comparison)
    """
    T = len(recording_observations)  # Number of timesteps in this recording
    N = hmm.states  # Number of states
    
    # Convert HMM matrices from linear to log space
    log_a_matrix = safe_log(hmm.A)  # Transition matrix in log space
    log_b_matrix = safe_log(hmm.B)  # Emission matrix in log space
    log_pi = safe_log(hmm.Pi)       # Initial probabilities in log space
    
    # Initialize log_alpha matrix in log space
    log_alpha = np.full((N, T), float('-inf'))
    
    # Initialize first timestep (t=0) in log space
    obs_symbol = recording_observations[0]
    for state in range(N):
        if log_pi[state] != float('-inf') and log_b_matrix[state, obs_symbol] != float('-inf'):
            log_alpha[state, 0] = log_pi[state] + log_b_matrix[state, obs_symbol]
        else:
            log_alpha[state, 0] = float('-inf')
    
    # Wrap single recording for calculate_log_alpha interface
    observations_list = [recording_observations]
    recording_idx = 0
    
    # Forward pass for remaining timesteps using calculate_log_alpha
    for timestep in range(1, T):
        for state in range(N):
            calculate_log_alpha(timestep, state, log_alpha, recording_idx, 
                              observations_list, log_a_matrix, log_b_matrix, N)
    
    # Calculate log P(O|lambda) for this recording
    # Sum the final alpha values in log space using log_sum_exp
    final_log_alphas = []
    for state in range(N):
        if log_alpha[state, T-1] != float('-inf'):
            final_log_alphas.append(log_alpha[state, T-1])
    
    if final_log_alphas:
        log_probability_O_given_lambda = log_sum_exp(np.array(final_log_alphas))
    else:
        log_probability_O_given_lambda = float('-inf')
    
    return log_probability_O_given_lambda


def test_hmm(all_hmm: List[HMMTrained], test_recordings_dict: Dict[str, List[List[RawDataMFCC]]], base_dir="../Data", show_progress=False) -> Tuple[List[str], List[str]]:
    """
    Test HMM models on test recordings and return predictions.
    
    Args:
        all_hmm: List of trained HMM models
        test_recordings_dict: Dictionary where keys are word names and values are lists of recordings
              Format: {"word1": [recording1_frames, recording2_frames, ...], "word2": [...], ...}
              Where each recording_frames is a list[RawDataMFCC]
              So the structure is: dict[str, list[list[RawDataMFCC]]]

    Returns:
        Tuple of (true_labels, predicted_labels)
    """
    print("Starting HMM testing...")

    storage = DataStorage()
    codevector_dir = os.path.join(base_dir, "CodeVector")
    centroids = storage.load_centroids(os.path.join(codevector_dir, "codevector.json"))
    
    # Phase 1: Calculate all observations (W calls to get_observations)
    print("Phase 1: Converting recordings to observations...")
    observations_dict = {}
    for word, recordings in test_recordings_dict.items():
        print(f"  Converting {len(recordings)} recordings for word: '{word}'")
        observations_dict[word] = get_observations(recordings, centroids)
    
    # Phase 2: Test all combinations
    print("Phase 2: Testing all recording-HMM combinations...")
    true_labels = []
    predicted_labels = []
    
    for true_word, observations_list in observations_dict.items():
        print(f"Testing {len(observations_list)} recordings for word: '{true_word}'")
        
        for recording_idx, single_recording_obs in enumerate(observations_list):
            best_likelihood = -float('inf')
            predicted_word = None
            
            likelihoods = {}
            for hmm in all_hmm:
                likelihood = calculate_log_likelihood(single_recording_obs, hmm)
                likelihoods[hmm.word] = likelihood
                
                if likelihood > best_likelihood:
                    best_likelihood = likelihood
                    predicted_word = hmm.word
            
            # Debug information
            if show_progress:
                print(f"  Recording {recording_idx+1} likelihoods: {likelihoods}")
                print(f"  True: '{true_word}' -> Predicted: '{predicted_word}'")
            
            true_labels.append(true_word)
            predicted_labels.append(predicted_word if predicted_word else "unknown")
    
    return true_labels, predicted_labels


def create_confusion_matrix(true_labels: List[str], predicted_labels: List[str], base_dir="../Data"):
    """
    Create and save a confusion matrix plot from test results.
    
    Args:
        true_labels: List of true word labels
        predicted_labels: List of predicted word labels
        base_dir: Base directory path
    """
    
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(base_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Get unique labels (words) and sort them for consistent ordering
    unique_labels = sorted(list(set(true_labels + predicted_labels)))
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=unique_labels)
    
    # Calculate accuracy
    accuracy = (cm.diagonal().sum() / cm.sum()) * 100
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=unique_labels,
                yticklabels=unique_labels,
                cbar_kws={'label': 'Number of Recordings'})
    
    plt.title(f'HMM Classification Confusion Matrix\nAccuracy: {accuracy:.2f}%', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Word', fontsize=12)
    plt.ylabel('True Word', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(plots_dir, "confusion_matrix.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels, labels=unique_labels))
    print(f"\nConfusion matrix saved to: {plot_path}")
    print(f"Overall Accuracy: {accuracy:.2f}%")
