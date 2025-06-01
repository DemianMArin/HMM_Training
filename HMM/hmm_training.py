import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from typing import List, Dict, Tuple
import sys
import logging
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hmm_classes import HMMTrained, DataStorageHMM
from CodeVector.codevector_classes import RawDataMFCC, CentroidDataMFCC 


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

logger = setup_logger(name="training")


def safe_log(x):
    """Safely compute log, returning -inf for values <= 0"""
    if isinstance(x, np.ndarray):
        result = np.full_like(x, float('-inf'), dtype=float)
        mask = x > 0
        result[mask] = np.log(x[mask])
        return result
    else:
        return math.log(x) if x > 0 else float('-inf')

def safe_exp(x):
    """Safely compute exp, handling -inf values"""
    if isinstance(x, np.ndarray):
        result = np.zeros_like(x, dtype=float)
        mask = x != float('-inf')
        result[mask] = np.exp(x[mask])
        return result
    else:
        return math.exp(x) if x != float('-inf') else 0.0

def log_sum_exp(log_probs):
    """Numerically stable log(sum(exp(log_probs))) for arrays or scalars"""
    if isinstance(log_probs, np.ndarray):
        # Remove -inf values for computation
        finite_mask = log_probs != float('-inf')
        if not np.any(finite_mask):
            return float('-inf')
        
        finite_probs = log_probs[finite_mask]
        max_val = np.max(finite_probs)
        return max_val + math.log(np.sum(np.exp(finite_probs - max_val)))
    else:
        # Handle scalar case
        return log_probs if log_probs != float('-inf') else float('-inf')


def get_observations(recordings: List[List[RawDataMFCC]], centroids: List[CentroidDataMFCC]) -> list[np.ndarray]:
    """
    Convert RawDataMFCC recordings to observation sequences using vector quantization.
    
    Args:
        recordings: List of List[RawDataMFCC objects] for one word
        centroids: List of CentroidDataMFCC objects (code vector)
    
    Returns:
        list[np.ndarray]: Array of centroid indices for each frame in each recording
    """
    observations = []

    for recording in recordings:
        recording_observations = []
        
        for frame in recording:
            # Extract MFCC features (excluding power - index 0)
            frame_mfcc = frame.mfcc[1:]  # Skip index 0 (power)
            
            min_distance = float('inf')
            closest_centroid_id = 0
            
            # Find closest centroid using Euclidean distance
            for centroid_idx, centroid in enumerate(centroids):
                centroid_mfcc = centroid.mfcc[1:]  # Skip index 0 (power)
                
                # Calculate Euclidean distance
                distance = np.linalg.norm(frame_mfcc - centroid_mfcc)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_centroid_id = centroid_idx
            
            recording_observations.append(closest_centroid_id)
        
        observations.append(np.array(recording_observations))
    
    return observations

def calculate_log_alpha(timestep: int, origin_state: int, log_alpha: np.ndarray, recording_idx: int, 
                       observations: List[np.ndarray], log_a_matrix: np.ndarray, log_b_matrix: np.ndarray, N: int):
    """
    Calculate forward variable log_alpha at given timestep and state.
    Pure log-space implementation for numerical stability.
    
    Args:
        timestep: Current time step
        origin_state: State we're calculating alpha for
        log_alpha: Log-space alpha matrix
        recording_idx: Index of current recording
        observations: List of observation sequences
        log_a_matrix: Log-space transition matrix
        log_b_matrix: Log-space emission matrix
        N: Number of states
    """
    # Sum over all previous states in log space
    log_sum_terms = []
    
    for prev_state in range(N):
        # Check if transition is possible (not -inf)
        if (log_alpha[prev_state, timestep-1] != float('-inf') and 
            log_a_matrix[prev_state, origin_state] != float('-inf')):
            
            log_term = log_alpha[prev_state, timestep-1] + log_a_matrix[prev_state, origin_state]
            log_sum_terms.append(log_term)
    
    if log_sum_terms:
        # Use log_sum_exp for numerical stability
        log_sum = log_sum_exp(np.array(log_sum_terms))
        
        # Add emission probability in log space
        obs_symbol = observations[recording_idx][timestep]
        if log_b_matrix[origin_state, obs_symbol] != float('-inf'):
            log_alpha[origin_state, timestep] = log_sum + log_b_matrix[origin_state, obs_symbol]
        else:
            log_alpha[origin_state, timestep] = float('-inf')
    else:
        log_alpha[origin_state, timestep] = float('-inf')


def calculate_log_beta(timestep: int, origin_state: int, log_beta: np.ndarray, recording_idx: int,
                      observations: List[np.ndarray], log_a_matrix: np.ndarray, log_b_matrix: np.ndarray, N: int):
    """
    Calculate backward variable log_beta at given timestep and state.
    Pure log-space implementation for numerical stability.
    
    Args:
        timestep: Current time step
        origin_state: State we're calculating beta for
        log_beta: Log-space beta matrix
        recording_idx: Index of current recording
        observations: List of observation sequences
        log_a_matrix: Log-space transition matrix
        log_b_matrix: Log-space emission matrix
        N: Number of states
    """
    # Sum over all next states in log space
    log_sum_terms = []
    
    for next_state in range(N):
        obs_symbol = observations[recording_idx][timestep + 1]
        
        # Check if transition is possible (not -inf)
        if (log_a_matrix[origin_state, next_state] != float('-inf') and 
            log_b_matrix[next_state, obs_symbol] != float('-inf') and
            log_beta[next_state, timestep + 1] != float('-inf')):
            
            log_term = (log_a_matrix[origin_state, next_state] + 
                       log_b_matrix[next_state, obs_symbol] + 
                       log_beta[next_state, timestep + 1])
            log_sum_terms.append(log_term)
    
    if log_sum_terms:
        # Use log_sum_exp for numerical stability
        log_beta[origin_state, timestep] = log_sum_exp(np.array(log_sum_terms))
    else:
        log_beta[origin_state, timestep] = float('-inf')

# Update the training function to save HMM models
def training_with_save(word_recordings: List[RawDataMFCC], centroids: List[CentroidDataMFCC], word_name: str, max_iterations = 100, show_progress=True) -> HMMTrained:
    """
    Main training function that coordinates the HMM training process and saves the model.
    
    Args:
        word_recordings: List of RawDataMFCC objects for one word
        centroids: List of CentroidDataMFCC objects (code vector)
        word_name: Name of the word being trained
    
    Returns:
        HMMTrained: Trained HMM model
    """
    print("Converting recordings to observations...")
    observations = get_observations(word_recordings, centroids)
    
    print(f"Generated {len(observations)} observation sequences")
    print(f"Sequence lengths: {[len(obs) for obs in observations]}")
    
    print("Starting Baum-Welch training...")
    A, B, pi = hmm_training(observations, N=4, M=len(centroids), max_iterations = max_iterations, show_progress=show_progress)
    
    # Create HMM model object
    hmm_model = HMMTrained(
        states=4,
        symbols=len(centroids),
        A=A,
        B=B,
        Pi=pi,
        word=word_name
    )
    
    # Save the model
    DataStorageHMM.save_hmm(hmm_model,print_messages=False)
    
    return hmm_model


def hmm_training(observations: List[np.ndarray], N: int = 4, M: int = 256, 
                epsilon: float = 1e-6, max_iterations: int = 100, show_progress = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Train HMM using Baum-Welch algorithm with proper log-space arithmetic.
    
    Args:
        observations: List of observation sequences (each is np.ndarray of centroid indices)
        N: Number of states
        M: Number of symbols (centroids)
        epsilon: Convergence threshold
        max_iterations: Maximum number of iterations
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (A, B, pi) matrices
    """
    num_recordings = len(observations)
    
    # Initialize parameters as specified in pseudo-code
    # pi_matrix = np.array([0.97, 0.02, 0.005, 0.005])
    pi_matrix = np.array([0.99, 0.006, 0.002, 0.002])

    
    # Transition matrix A (left-to-right topology)
    # a_matrix = np.array([
    #     [0.6, 0.4, 0.0, 0.0],
    #     [0.0, 0.6, 0.4, 0.0],
    #     [0.0, 0.0, 0.6, 0.4],
    #     [0.0, 0.0, 0.0, 1.0]
    # ])
    # a_matrix = np.array([
    #     [0.7, 0.3, 0.0, 0.0],
    #     [0.0, 0.7, 0.3, 0.0],
    #     [0.0, 0.0, 0.7, 0.3],
    #     [0.0, 0.0, 0.0, 1.0]
    # ])
    a_matrix = np.array([
        [0.555, 0.445, 0.0, 0.0],
        [0.0, 0.555, 0.445, 0.0],
        [0.0, 0.0, 0.555, 0.445],
        [0.0, 0.0, 0.0, 1.0]
    ])

    
    # Emission matrix B (uniform initialization)
    b_matrix = np.full((N, M), 1.0/M)
    
    # Convert to log space for numerical stability
    log_pi_matrix = safe_log(pi_matrix)
    log_a_matrix = safe_log(a_matrix)
    log_b_matrix = safe_log(b_matrix)
    
    # Pre-allocate arrays for forward-backward variables (in log space)
    log_alpha_list = []
    log_beta_list = []
    log_gamma_list = []
    log_xi_list = []
    log_probability_O_given_lambda = np.full(num_recordings, float('-inf'))
    
    for i in range(num_recordings):
        T = len(observations[i])
        log_alpha_list.append(np.full((N, T), float('-inf')))
        log_beta_list.append(np.full((N, T), float('-inf')))
        log_gamma_list.append(np.full((N, T), float('-inf')))
        log_xi_list.append(np.full((N, N, T-1), float('-inf')))
    
    # Training loop
    prev_log_likelihood_sum = float('-inf')
    diff = epsilon + 10
    
    iteration = 0
    while diff >= epsilon and iteration < max_iterations:
        if show_progress:
            print(f"Iteration {iteration + 1}")
        
        # Forward-Backward Variables calculation
        for recording in range(num_recordings):
            log_alpha = log_alpha_list[recording]
            log_beta = log_beta_list[recording]
            total_time = len(observations[recording])
            
            # Initialize log_alpha at t=0
            for state in range(N):
                if total_time > 0:
                    obs_symbol = observations[recording][0]
                    log_alpha[state, 0] = log_pi_matrix[state] + log_b_matrix[state, obs_symbol]
                    
                    # Initialize log_beta at t=T-1
                    log_beta[state, total_time-1] = 0.0  # log(1) = 0
            
            # Forward pass
            for timestep in range(1, total_time):
                for state in range(N):
                    calculate_log_alpha(timestep, state, log_alpha, recording, observations, log_a_matrix, log_b_matrix, N)
            
            # Backward pass
            for timestep in range(total_time - 2, -1, -1):
                for state in range(N):
                    calculate_log_beta(timestep, state, log_beta, recording, observations, log_a_matrix, log_b_matrix, N)
            
            # Calculate log P(O|lambda) for this recording
            log_terms = log_alpha[:, total_time-1]
            log_probability_O_given_lambda[recording] = log_sum_exp(log_terms)
        
        # Expectation Variables (log_gamma and log_xi)
        for recording in range(num_recordings):
            log_alpha = log_alpha_list[recording]
            log_beta = log_beta_list[recording]
            log_gamma = log_gamma_list[recording]
            log_xi = log_xi_list[recording]
            total_time = len(observations[recording])
            log_prob_O = log_probability_O_given_lambda[recording]
            
            # Calculate log_gamma
            for timestep in range(total_time):
                for state in range(N):
                    if log_prob_O != float('-inf'):
                        log_gamma[state, timestep] = log_alpha[state, timestep] + log_beta[state, timestep] - log_prob_O
                    else:
                        log_gamma[state, timestep] = float('-inf')
            
            # Calculate log_xi
            for timestep in range(total_time - 1):
                obs_symbol = observations[recording][timestep + 1]
                for state in range(N):
                    for next_state in range(N):
                        if log_prob_O != float('-inf'):
                            log_xi[state, next_state, timestep] = (
                                log_alpha[state, timestep] + 
                                log_a_matrix[state, next_state] + 
                                log_b_matrix[next_state, obs_symbol] + 
                                log_beta[next_state, timestep + 1] - 
                                log_prob_O
                            )
                        else:
                            log_xi[state, next_state, timestep] = float('-inf')
        
        # Maximization step - Parameter re-estimation
        
        # Update Pi (initial state probabilities)
        log_pi_new = np.full(N, float('-inf'))
        for state in range(N):
            log_terms = []
            for recording in range(num_recordings):
                log_gamma = log_gamma_list[recording]
                if log_gamma[state, 0] != float('-inf'):
                    log_terms.append(log_gamma[state, 0])
            
            if log_terms:
                log_pi_new[state] = log_sum_exp(np.array(log_terms)) - math.log(num_recordings)
        
        log_pi_matrix = log_pi_new
        
        # Update A (transition matrix)
        log_a_new = np.full((N, N), float('-inf'))
        for origin_state in range(N):
            # Calculate denominator (sum of gamma for origin_state, excluding last timestep)
            log_denom_terms = []
            for recording in range(num_recordings):
                log_gamma = log_gamma_list[recording]
                total_time = log_gamma.shape[1]
                for timestep in range(total_time - 1):  # Exclude last timestep
                    if log_gamma[origin_state, timestep] != float('-inf'):
                        log_denom_terms.append(log_gamma[origin_state, timestep])
            
            if log_denom_terms:
                log_denominator = log_sum_exp(np.array(log_denom_terms))
                
                for destination_state in range(N):
                    # Calculate numerator (sum of xi for this state transition)
                    log_num_terms = []
                    for recording in range(num_recordings):
                        log_xi = log_xi_list[recording]
                        total_time = log_xi.shape[2]
                        for timestep in range(total_time):
                            if log_xi[origin_state, destination_state, timestep] != float('-inf'):
                                log_num_terms.append(log_xi[origin_state, destination_state, timestep])
                    
                    if log_num_terms:
                        log_numerator = log_sum_exp(np.array(log_num_terms))
                        log_a_new[origin_state, destination_state] = log_numerator - log_denominator
        
        log_a_matrix = log_a_new
        
        # Update B (emission matrix)
        log_b_new = np.full((N, M), float('-inf'))
        for state in range(N):
            # Calculate denominator (sum of all gamma for this state)
            log_denom_terms = []
            for recording in range(num_recordings):
                log_gamma = log_gamma_list[recording]
                total_time = log_gamma.shape[1]
                for timestep in range(total_time):
                    if log_gamma[state, timestep] != float('-inf'):
                        log_denom_terms.append(log_gamma[state, timestep])
            
            if log_denom_terms:
                log_denominator = log_sum_exp(np.array(log_denom_terms))
                
                for symbol in range(M):
                    # Calculate numerator (sum of gamma when observing this symbol)
                    log_num_terms = []
                    for recording in range(num_recordings):
                        log_gamma = log_gamma_list[recording]
                        total_time = log_gamma.shape[1]
                        for timestep in range(total_time):
                            if (observations[recording][timestep] == symbol and 
                                log_gamma[state, timestep] != float('-inf')):
                                log_num_terms.append(log_gamma[state, timestep])
                    
                    if log_num_terms:
                        log_numerator = log_sum_exp(np.array(log_num_terms))
                        log_b_new[state, symbol] = log_numerator - log_denominator
                    else:
                        log_b_new[state, symbol] = safe_log(1e-10)  # Small value to avoid -inf
        
        log_b_matrix = log_b_new
        
        # Check convergence
        current_log_likelihood_sum = log_sum_exp(log_probability_O_given_lambda)
        
        if prev_log_likelihood_sum != float('-inf'):
            diff = abs(current_log_likelihood_sum - prev_log_likelihood_sum)
        else:
            diff = float('inf')
        
        if show_progress:
            print(f"Log-likelihood: {current_log_likelihood_sum:.6f}, Diff: {diff:.8f}")
        
        prev_log_likelihood_sum = current_log_likelihood_sum
        iteration += 1
    
    if iteration >= max_iterations:
        print(f"Log-likelihood: {current_log_likelihood_sum:.6f}, Diff: {diff:.8f}")
        print(f"Reached maximum iterations ({max_iterations})")
    else:
        print(f"Log-likelihood: {current_log_likelihood_sum:.6f}, Diff: {diff:.8f}")
        print(f"Converged after {iteration} iterations")
    
    # Convert back to linear space for return
    pi_matrix = safe_exp(log_pi_matrix)
    a_matrix = safe_exp(log_a_matrix)
    b_matrix = safe_exp(log_b_matrix)
    
    # Normalize to ensure probabilities sum to 1 (handle any numerical errors)
    pi_matrix = pi_matrix / np.sum(pi_matrix)
    
    for i in range(N):
        row_sum = np.sum(a_matrix[i, :])
        if row_sum > 0:
            a_matrix[i, :] = a_matrix[i, :] / row_sum
    
    for i in range(N):
        row_sum = np.sum(b_matrix[i, :])
        if row_sum > 0:
            b_matrix[i, :] = b_matrix[i, :] / row_sum
    
    return a_matrix, b_matrix, pi_matrix
