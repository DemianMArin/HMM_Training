"""
codevector_classes.py
Data classes for the codevector creation system.
"""

import numpy as np
import librosa
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple
import json
import pickle
import warnings
import logging
import sys

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

logger = setup_logger(name="classes")

# Try to import spectrum for LSF conversion
try:
    from spectrum import poly2lsf, lsf2poly
except ImportError:
    warnings.warn("spectrum package not found. Using alternative LSF conversion.")
    # Alternative implementation if spectrum is not available
    def poly2lsf(poly):
        """Convert polynomial coefficients to line spectral frequencies."""
        # This is a simplified version - for production use, install spectrum package
        # pip install spectrum
        order = len(poly) - 1
        lsf = np.zeros(order)
        # Placeholder implementation - replace with actual conversion
        angles = np.arccos(np.clip(-poly[1:] / 2, -1, 1))
        lsf = angles / np.pi
        return lsf
    
    def lsf2poly(lsf):
        """Convert line spectral frequencies to polynomial coefficients."""
        # Placeholder implementation
        order = len(lsf)
        poly = np.zeros(order + 1)
        poly[0] = 1
        poly[1:] = -2 * np.cos(lsf * np.pi)
        return poly


    def verify_calculations(self):
        """Verify that frame calculations are correct."""
        issues = []
        
        # Check raw samples
        if len(self.raw_samples) == 0:
            issues.append("no raw samples")
        
        # Check autocorrelation
        if not np.any(self.autocorrelation_vector):
            issues.append("autocorrelation is all zeros")
        elif len(self.autocorrelation_vector) != 13:
            issues.append(f"autocorrelation has wrong length: {len(self.autocorrelation_vector)}")
        
        # Check LPC
        if not np.any(self.lpc_vector):
            issues.append("lpc is all zeros")
        elif len(self.lpc_vector) != 13:
            issues.append(f"lpc has wrong length: {len(self.lpc_vector)}")
        elif self.lpc_vector[0] != 1.0:
            issues.append(f"lpc[0] should be 1.0, got {self.lpc_vector[0]}")
        
        # Check LSF
        if not np.any(self.lsf_vector):
            issues.append("lsf is all zeros")
        elif len(self.lsf_vector) != 12:
            issues.append(f"lsf has wrong length: {len(self.lsf_vector)}")
        
        return issues


@dataclass
class RawDataTraining:
    """Class to store frame data for HMM training and testing (simplified version)."""
    raw_samples: np.ndarray
    autocorrelation_vector: np.ndarray = field(default_factory=lambda: np.zeros(13))
    frame_number: int = 0
    recording: str = ""
    
    def __post_init__(self):
        """Calculate autocorrelation upon initialization."""
        if len(self.raw_samples) > 0:
            self.calculate_autocorrelation()
    
    def calculate_autocorrelation(self):
        """Calculate autocorrelation using librosa."""
        # Ensure we have enough samples (at least order + 1)
        if len(self.raw_samples) > 12:
            self.autocorrelation_vector = librosa.autocorrelate(
                self.raw_samples.astype(float), 
                max_size=13
            )[:13]
        else:
            # For very short frames, use what we have
            max_size = min(13, len(self.raw_samples))
            self.autocorrelation_vector[:max_size] = librosa.autocorrelate(
                self.raw_samples.astype(float), 
                max_size=max_size
            )
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'raw_samples': self.raw_samples.tolist(),
            'autocorrelation_vector': self.autocorrelation_vector.tolist(),
            'frame_number': self.frame_number,
            'recording': self.recording
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create RawDataTraining from dictionary."""
        return cls(
            raw_samples=np.array(data['raw_samples']),
            autocorrelation_vector=np.array(data['autocorrelation_vector']),
            frame_number=data['frame_number'],
            recording=data['recording']
        )


@dataclass
class RawData:
    """Class to store individual frame data from audio recordings."""
    raw_samples: np.ndarray
    autocorrelation_vector: np.ndarray = field(default_factory=lambda: np.zeros(13))
    lpc_vector: np.ndarray = field(default_factory=lambda: np.zeros(13))
    lsf_vector: np.ndarray = field(default_factory=lambda: np.zeros(12))
    parent_centroid_id: int = 0
    generation: int = 0
    frame_number: int = 0
    recording: str = ""
    
    def __post_init__(self):
        """Calculate autocorrelation, LPC, and LSF upon initialization."""
        if len(self.raw_samples) > 0:
            self.calculate_autocorrelation()
            self.calculate_lpc()
            self.calculate_lsf()

            self.raw_samples = self.raw_samples.flatten()
            self.autocorrelation_vector = self.autocorrelation_vector.flatten()
            self.lpc_vector = self.lpc_vector.flatten()
            self.lsf_vector= self.lsf_vector.flatten()
  
    def calculate_autocorrelation(self):
        """Calculate autocorrelation using librosa."""
        # Ensure we have enough samples (at least order + 1)
        if len(self.raw_samples) > 12:
            self.autocorrelation_vector = librosa.autocorrelate(
                self.raw_samples.astype(float), 
                max_size=13
            )[:13]
        else:
            # For very short frames, use what we have
            max_size = min(13, len(self.raw_samples))
            self.autocorrelation_vector[:max_size] = librosa.autocorrelate(
                self.raw_samples.astype(float), 
                max_size=max_size
            )
    
    def calculate_lpc(self):
        """Calculate LPC coefficients using librosa."""
        try:
            if len(self.raw_samples) > 12:
                # librosa.lpc returns a 1D array of size order+1 (13 for order 12)
                self.lpc_vector = librosa.lpc(self.raw_samples.astype(float).flatten(), order=12)
            else:
                # Use lower order for short frames
                order = min(12, len(self.raw_samples) - 1)
                if order > 0:
                    lpc_coeffs = librosa.lpc(self.raw_samples.astype(float), order=order)
                    # Copy the coefficients we got and pad with zeros if needed
                    self.lpc_vector[:len(lpc_coeffs)] = lpc_coeffs
                    # Ensure first coefficient is 1.0 if we had to pad
                    if len(lpc_coeffs) < 13:
                        self.lpc_vector[0] = 1.0
                else:
                    # Fallback for very short frames
                    self.lpc_vector[0] = 1.0
        except Exception as e:
            # If LPC calculation fails, use default values
            self.lpc_vector[0] = 1.0
            print(f"Warning: LPC calculation failed for frame, using defaults: {e}")
    
    def calculate_lsf(self):
        """Convert LPC to LSF (Line Spectral Frequencies)."""
        try:
            # Convert LPC coefficients to LSF
            lsf = np.array(poly2lsf(self.lpc_vector))
            
            # Store the LSF coefficients (should be 12 for order 12)
            if len(lsf) >= 12:
                self.lsf_vector = lsf
            else:
                self.lsf_vector[:len(lsf)] = lsf

        except Exception as e:
            # If LSF conversion fails, keep defaults (zeros)
            print(f"Warning: LSF conversion failed for frame, using defaults: {e}")
            pass
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'raw_samples': self.raw_samples.tolist(),
            'autocorrelation_vector': self.autocorrelation_vector.tolist(),
            'lpc_vector': self.lpc_vector.tolist(),
            'lsf_vector': self.lsf_vector.tolist(),
            'parent_centroid_id': self.parent_centroid_id,
            'generation': self.generation,
            'frame_number': self.frame_number,
            'recording': self.recording
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create RawData from dictionary."""
        return cls(
            raw_samples=np.array(data['raw_samples']),
            autocorrelation_vector=np.array(data['autocorrelation_vector']),
            lpc_vector=np.array(data['lpc_vector']),
            lsf_vector=np.array(data['lsf_vector']),
            parent_centroid_id=data['parent_centroid_id'],
            generation=data['generation'],
            frame_number=data['frame_number'],
            recording=data['recording']
        )

@dataclass
class RawDataMFCC:
    """Class to store individual frame data with MFCC coefficients from audio recordings."""
    raw_samples: np.ndarray
    sample_rate: int = 16000  # 16kHz default
    n_channels: int = 1       # Mono default
    frame_duration_ms: float = 20.0  # 20ms frame default
    mfcc: np.ndarray = field(default_factory=lambda: np.zeros(13))
    parent_centroid_id: int = 0
    generation: int = 0
    frame_number: int = 0
    recording: str = ""
    
    def __post_init__(self):
        """Calculate MFCC coefficients upon initialization."""
        if len(self.raw_samples) > 0:
            self.calculate_mfcc()
            
            # Ensure all arrays are flattened for consistency
            self.raw_samples = self.raw_samples.flatten()
            self.mfcc= self.mfcc.flatten()
    
    def calculate_mfcc(self):
        """Calculate MFCC coefficients using librosa."""
        try:

            # Use frame length for n_fft since its single frames
            frame_length = len(self.raw_samples)

            # Calculate MFCC for single frame
            mfcc_features = librosa.feature.mfcc(
                y=self.raw_samples.astype(float).flatten(),
                sr=self.sample_rate,
                n_mfcc=13,
                n_fft=frame_length,  
                hop_length=None,  # Process entire frame at once
                center=False,  # Don't pad the frame
                n_mels=26  # Standard number of mel filters
            )
            
            # Extract the MFCC coefficients (should be shape (13, 1))
            self.mfcc= mfcc_features.flatten()
            
        except Exception as e:
            # If MFCC calculation fails, keep default zeros
            print(f"Warning: MFCC calculation failed for frame, using defaults: {e}")
            self.mfcc= np.zeros(13)
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'raw_samples': self.raw_samples.tolist(),
            'sample_rate': self.sample_rate,
            'n_channels': self.n_channels,
            'frame_duration_ms': self.frame_duration_ms,
            'mfcc_vector': self.mfcc.tolist(),
            'parent_centroid_id': self.parent_centroid_id,
            'generation': self.generation,
            'frame_number': self.frame_number,
            'recording': self.recording
        }

    @classmethod
    def from_dict(cls, data):
        """Create RawDataMFCC from dictionary."""
        return cls(
            raw_samples=np.array(data['raw_samples']),
            sample_rate=data['sample_rate'],
            n_channels=data['n_channels'],
            frame_duration_ms=data['frame_duration_ms'],
            mfcc=np.array(data['mfcc_vector']),
            parent_centroid_id=data['parent_centroid_id'],
            generation=data['generation'],
            frame_number=data['frame_number'],
            recording=data['recording']
        )

@dataclass
class CentroidData:
    """Class to store centroid information."""
    lsf: np.ndarray
    lpc: np.ndarray = field(default_factory=lambda: np.zeros(13))
    id: int = 0
    
    def __post_init__(self):
        """Calculate LPC from LSF upon initialization."""
        if len(self.lsf) == 12 and not np.any(self.lpc):
            self.calculate_lpc()
    
    def calculate_lpc(self):
        """Convert LSF to LPC coefficients."""
        if len(self.lsf) == 12:
            try:
                self.lpc = lsf2poly(self.lsf)

            except Exception as e:
                # If conversion fails, use default
                logger.error(f"Error lsf2poly conversion failed: {e}")
                pass
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'lsf': self.lsf.tolist(),
            'lpc': self.lpc.tolist(),
            'id': self.id
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create CentroidData from dictionary."""
        return cls(
            lsf=np.array(data['lsf']),
            lpc=np.array(data['lpc']),
            id=data['id']
        )

@dataclass
class CentroidDataMFCC:
    mfcc: np.ndarray = field(default_factory=lambda: np.zeros(13))
    id: int = 0

    def __post_init__(self):
        self.mfcc.flatten()

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'mfcc': self.mfcc.tolist(),
            'id': self.id
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create CentroidData from dictionary."""
        return cls(
            mfcc=np.array(data['mfcc']),
            id=data['id']
        )



class AudioProcessor:
    """Class to handle audio processing and frame extraction with overlap."""
    
    def __init__(self, sample_rate=16000, frame_duration_ms=20, overlap_ms=10):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.overlap_ms = overlap_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)  # 320 samples
        self.overlap_size = int(sample_rate * overlap_ms / 1000)  # 160 samples
        self.hop_size = self.frame_size - self.overlap_size  # 160 samples
    
    def process_recording(self, audio_path: str, purpose: str):
        """Process a single audio recording into frames for a specific purpose.
        
        Args:
            audio_path: Path to the .npy audio file
            purpose: One of 'train', 'hmm', or 'test'
        
        Returns:
            List of RawData (for train) or RawDataTraining (for hmm/test) frames
        """
        # Load audio
        audio_data = np.load(audio_path)
        
        # Split audio into frames with overlap
        frames = self._split_into_frames_with_overlap(audio_data)
        
        # Create appropriate data objects based on purpose
        raw_data_list = []
        recording_name = Path(audio_path).stem  # e.g., "finish-01"
        
        # Before different classes were used for diferent purposes
        for i, frame in enumerate(frames):
            if purpose == 'train':
                # Use full RawData class for training (with LPC, LSF calculations)
                # raw_data = RawData(
                #     raw_samples=frame,
                #     frame_number=i,
                #     recording=recording_name
                # )

                # Use RawDataMFCC. Better representaion of voice
                raw_data = RawDataMFCC(
                    raw_samples=frame,
                    frame_number=i,
                    recording=recording_name
                )
            else:  # purpose == 'hmm' or purpose == 'test'

                # Use simplified RawDataTraining class (autocorrelation only)
                # raw_data = RawDataTraining(
                #     raw_samples=frame,
                #     frame_number=i,
                #     recording=recording_name
                # )

                # Use RawDataMFCC. Better representaion of voice
                raw_data = RawDataMFCC(
                    raw_samples=frame,
                    frame_number=i,
                    recording=recording_name
                )

            raw_data_list.append(raw_data)
        
        return raw_data_list
    
    def _split_into_frames_with_overlap(self, audio_data: np.ndarray) -> List[np.ndarray]:
        """Split audio data into overlapping frames."""
        frames = []
        
        # Use hop_size for overlap
        for i in range(0, len(audio_data) - self.frame_size + 1, self.hop_size):
            frame = audio_data[i:i + self.frame_size]
            frames.append(frame)
        
        # Handle the last frame if there are remaining samples
        # No padding - just use whatever samples remain
        last_start = len(frames) * self.hop_size
        if last_start < len(audio_data):
            last_frame = audio_data[last_start:]
            # Only add if it has enough samples to be meaningful (e.g., > 12 for LPC order 12)
            if len(last_frame) > 12:
                frames.append(last_frame)
        
        return frames


class DataStorage:
    """Class to handle saving and loading data."""
    
    @staticmethod
    def save_raw_data(raw_data_list, filepath: str, print_messages=False):
        """Save raw data to JSON file (works with both RawData, RawDataTraining, RawDataMFCC)."""
        data = [frame.to_dict() for frame in raw_data_list]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        if print_messages:
            print(f"    Saved {len(raw_data_list)} frames to {filepath}")
    
    @staticmethod
    def load_raw_data(filepath: str, data_type: str = 'auto', print_messages = True):
        """Load raw data from JSON file.
        
        Args:
            filepath: Path to JSON file
            data_type: 'train' for RawData, 'hmm'/'test' for RawDataTraining, 'auto' to detect
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if not data:  # Empty file
            return []
        
        # Auto-detect type based on data structure
        if data_type == 'auto':
            sample_item = data[0]
            if 'lpc_vector' in sample_item and 'lsf_vector' in sample_item:
                data_type = 'train'
            else:
                data_type = 'hmm'  # or 'test', same class
        
        if data_type == 'train':
            raw_data_list = [RawData.from_dict(frame_data) for frame_data in data]
        else:  # 'hmm' or 'test'
            raw_data_list = [RawDataTraining.from_dict(frame_data) for frame_data in data]
        
        if print_messages:
            print(f"  Loaded {len(raw_data_list)} frames from {filepath}")
        return raw_data_list

    @staticmethod
    def load_raw_data_mfcc(filepath: str, data_type: str = 'auto', print_messages = True):
        """Load raw data from JSON file.
        
        Args:
            filepath: Path to JSON file
            data_type: 'train' , 'hmm'/'test' , 'auto' to detect
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if not data:  # Empty file
            return []
        
        raw_data_list = [RawDataMFCC.from_dict(frame_data) for frame_data in data]
        
        if print_messages:
            print(f"  Loaded {len(raw_data_list)} frames from {filepath}")
        return raw_data_list

    @staticmethod
    def save_centroids(centroids: List[CentroidData], filepath: str):
        """Save centroids to JSON file."""
        data = [centroid.to_dict() for centroid in centroids]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(centroids)} centroids to {filepath}")
    
    @staticmethod
    def load_centroids(filepath: str, print_messages=True) -> List[CentroidData]:
        """Load centroids from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        centroids = [CentroidData.from_dict(centroid_data) for centroid_data in data]
        if print_messages:
            print(f"Loaded {len(centroids)} centroids from {filepath}")
        return centroids
    
    @staticmethod
    def save_generations(generations: List[List[CentroidData]], filepath: str):
        """Save all generations to a single JSON file."""
        data = []
        for gen_centroids in generations:
            gen_data = [centroid.to_dict() for centroid in gen_centroids]
            data.append(gen_data)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(generations)} generations to {filepath}")
    
    @staticmethod
    def load_generations(filepath: str) -> List[List[CentroidData]]:
        """Load generations from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        generations = []
        for gen_data in data:
            gen_centroids = [CentroidData.from_dict(centroid_data) for centroid_data in gen_data]
            generations.append(gen_centroids)
        
        print(f"Loaded {len(generations)} generations from {filepath}")
        return generations

    @staticmethod
    def save_centroids(centroids: List[CentroidDataMFCC], filepath: str):
        """Save centroids to JSON file."""
        data = [centroid.to_dict() for centroid in centroids]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(centroids)} centroids to {filepath}")
    
    @staticmethod
    def load_centroids(filepath: str, print_messages=True) -> List[CentroidDataMFCC]:
        """Load centroids from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        centroids = [CentroidDataMFCC.from_dict(centroid_data) for centroid_data in data]
        if print_messages:
            print(f"Loaded {len(centroids)} centroids from {filepath}")
        return centroids
    
    @staticmethod
    def save_generations(generations: List[List[CentroidDataMFCC]], filepath: str):
        """Save all generations to a single JSON file."""
        data = []
        for gen_centroids in generations:
            gen_data = [centroid.to_dict() for centroid in gen_centroids]
            data.append(gen_data)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(generations)} generations to {filepath}")
    
    @staticmethod
    def load_generations(filepath: str) -> List[List[CentroidDataMFCC]]:
        """Load generations from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        generations = []
        for gen_data in data:
            gen_centroids = [CentroidDataMFCC.from_dict(centroid_data) for centroid_data in gen_data]
            generations.append(gen_centroids)
        
        print(f"Loaded {len(generations)} generations from {filepath}")
        return generations
    
    @staticmethod
    def save_data_binary(data, filepath: str, print_messages =False):
        """Save data using pickle for faster loading (alternative to JSON)."""
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        if print_messages:
            print(f"    Saved data to {filepath} (binary format)")
    
    @staticmethod
    def load_data_binary(filepath: str):
        """Load data from pickle file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded data from {filepath} (binary format)")
        return data
