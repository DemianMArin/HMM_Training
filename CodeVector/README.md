# Codevector Creation System

This system processes audio recordings individually and creates codevectors for speech recognition applications. Each recording is processed separately to maintain clear separation between different samples.

## Project Structure

```
Project/
├── CodeVector/                    # Source code directory
│   ├── codevector_classes.py     # Data classes and utilities
│   ├── codevector_functions.py   # Core algorithms
│   ├── main.py                   # Main execution script
│   └── README.md                 # This file
├── Data/                         # Data directory
│   ├── Processed/                # Input processed audio files
│   │   ├── word1/                # Word category 1
│   │   │   ├── word1-01.npy     # Individual recordings
│   │   │   ├── word1-02.npy
│   │   │   └── ...
│   │   ├── word2/                # Word category 2
│   │   │   ├── word2-01.npy
│   │   │   └── ...
│   │   └── ...
│   ├── CodeVector/               # Training data and codevector
│   │   ├── codevector.json      # Generated codevector
│   │   ├── codevector.pkl
│   │   ├── generations.json
│   │   ├── generations.pkl
│   │   ├── codevector_frames.json # All training frames combined
│   │   └── codevector_frames.pkl
│   ├── TrainHMM/                 # HMM training data
│   │   ├── word1/
│   │   │   ├── word1-02/         # HMM recording
│   │   │   │   ├── hmm_frames.json
│   │   │   │   └── hmm_frames.pkl
│   │   │   └── ...
│   │   ├── word2/
│   │   └── ...
│   └── Test/                     # Test data
│       ├── word1/
│       │   ├── word1-03/         # Test recording
│       │   │   ├── test_frames.json
│       │   │   └── test_frames.pkl
│       │   └── ...
│       ├── word2/
│       └── ...
│   └── ...
```

## Workflow

### 1. Recording Purpose Assignment
- Each entire recording is assigned to ONE specific purpose:
  - **70% of recordings** → Codevector training
  - **20% of recordings** → HMM training
  - **10% of recordings** → Testing
- No frame-level splitting within recordings
- Assignment is done per word category to maintain balance

### 2. Individual Recording Processing
- Each `.npy` file is processed into overlapping frames (20ms frames, 10ms overlap)
- ALL frames from each recording serve the assigned purpose
- Recording metadata is preserved in each frame

### 3. Codevector Creation
- Only frames from TRAINING recordings are used
- LBG (Linde-Buzo-Gray) algorithm creates the codevector
- Default: 256 centroids with epsilon = 0.01
- Codevector represents the training data only

### 4. Data Organization
- Each recording maintains a directory with purpose-specific files
- Non-relevant purpose files are created empty for consistency
- Both JSON (human-readable) and pickle (fast loading) formats

## Key Features

### Frame Processing
- **Sample Rate**: 16 kHz
- **Frame Duration**: 20ms (320 samples)
- **Overlap**: 10ms (160 samples)
- **Hop Size**: 160 samples

### Signal Analysis
- Autocorrelation calculation (13 coefficients)
- Linear Predictive Coding (LPC) analysis (12th order)
- Line Spectral Frequencies (LSF) conversion
- Itakura-Saito distance metric for clustering

### Data Classes
- **RawData**: Full analysis for training frames (autocorrelation + LPC + LSF)
- **RawDataTraining**: Simplified for HMM/test frames (autocorrelation only)
- **CentroidData**: Codevector centroids with LSF/LPC representations
- **AudioProcessor**: Handles frame extraction and overlap
- **DataStorage**: Manages saving/loading in multiple formats

## Usage

```python
# Run the complete workflow
python main.py

# This will:
# 1. Process all recordings in Data/Processed/
# 2. Create individual train/test/HMM splits per recording
# 3. Generate shared codevector from all training data
# 4. **Scalable Processing**: Easy to add new recordings or word categories
5. **Multiple File Formats**: JSON for inspection, pickle for performance
6. **Clear Data Lineage**: Know exactly which recording produced each frame

## Advanced Usage

### Loading Specific Recording Data

```python
from codevector_classes import DataStorage

storage = DataStorage()

# Load all training frames (combined)
train_frames = storage.load_raw_data("Data/CodeVector/codevector_frames.json", "train")

# Load specific HMM recording
hmm_frames = storage.load_raw_data("Data/TrainHMM/finish/finish-02/hmm_frames.json", "hmm")

# Load specific test recording
test_frames = storage.load_raw_data("Data/Test/finish/finish-03/test_frames.json", "test")

# Load codevector
centroids = storage.load_centroids("Data/CodeVector/codevector.json")
```

### Processing Custom Recordings

```python
from codevector_classes import AudioProcessor

processor = AudioProcessor(sample_rate=16000, frame_duration_ms=20, overlap_ms=10)

# Process single recording with custom splits
train, hmm, test = processor.process_recording(
    "path/to/recording.npy",
    train_percent=0.8,  # 80% training
    hmm_percent=0.15,   # 15% HMM
    test_percent=0.05   # 5% testing
)
```

### Creating Custom Codevectors

```python
from codevector_functions import createCodeVector

# Create codevector with different parameters
centroids, generations = createCodeVector(
    raw_data_vocabulary=train_frames,
    centroids_quantity=512,  # More centroids
    epsilon=0.001           # Higher precision
)
```

## Troubleshooting

### Common Issues

1. **Missing spectrum package**: LSF conversion falls back to simplified version
   ```bash
   pip install spectrum
   ```

2. **Short audio files**: Minimum 12 samples needed for LPC analysis
   - Files shorter than frame_size are handled gracefully
   - Very short frames are filtered out automatically

3. **Memory usage**: Large datasets use significant RAM
   - Use pickle files for faster loading
   - Process recordings in batches if needed

4. **File permissions**: Ensure write access to Data/ directory
   ```bash
   chmod -R 755 Data/
   ```

### Performance Tips

1. **Use pickle files** for repeated loading (much faster than JSON)
2. **Adjust frame parameters** based on your audio characteristics
3. **Monitor memory usage** with large numbers of recordings
4. **Use appropriate epsilon** values for codevector convergence

## Algorithm Details

### LBG (Linde-Buzo-Gray) Algorithm
1. Start with single centroid (mean of all training data)
2. Split centroids using epsilon perturbation
3. Assign frames to nearest centroids using Itakura-Saito distance
4. Update centroids based on assigned frames
5. Repeat until convergence or max iterations
6. Continue splitting until desired number of centroids

### Itakura-Saito Distance
Used for measuring similarity between audio frames and centroids:
```
distance = auto_coeff[0] * lpc_centroid[0] + 2 * sum(auto_coeff[1:] * lpc_centroid[1:])
```

## Future Enhancements

- [ ] Add support for different audio formats (WAV, FLAC)
- [ ] Implement parallel processing for large datasets
- [ ] Add visualization tools for codevector analysis
- [ ] Support for dynamic frame sizing
- [ ] Integration with HMM training pipelines
- [ ] Cross-validation utilities for model evaluation

## Contributing

When modifying the code:
1. Maintain backward compatibility with existing data files
2. Update this README with any new features
3. Test with different audio file sizes and formats
4. Ensure proper error handling for edge cases

## License

This project is part of a speech recognition research system. Please refer to your institution's guidelines for usage and distribution.. Save results in organized structure
```

## Configuration

Key parameters can be modified in `main.py`:

```python
# Audio processing
sample_rate = 16000
frame_duration_ms = 20
overlap_ms = 10

# Data splitting
train_percent = 0.7  # 70% of recordings for training
hmm_percent = 0.2    # 20% of recordings for HMM training  
test_percent = 0.1   # 10% of recordings for testing

# Codevector creation
centroids_quantity = 256
epsilon = 0.01
```

## Dependencies

- `numpy`: Numerical computations
- `librosa`: Audio signal processing
- `pathlib`: File system operations  
- `json`: Data serialization
- `pickle`: Binary data storage
- `spectrum` (optional): Enhanced LSF conversion

Install missing dependencies:
```bash
pip install numpy librosa spectrum
```

## Output Files

### Per Recording
Data is organized by purpose first:
- **Training recordings**: All frames combined in `CodeVector/codevector_frames.json/pkl` (RawData objects)
- **HMM recordings**: Individual files in `TrainHMM/[word]/[recording]/hmm_frames.json/pkl` (RawDataTraining objects)
- **Test recordings**: Individual files in `Test/[word]/[recording]/test_frames.json/pkl` (RawDataTraining objects)

### Shared Codevector & Training Data
- `codevector.json/pkl`: Final centroids (from training recordings)
- `generations.json/pkl`: All generation data from LBG algorithm
- `codevector_frames.json/pkl`: All training frames combined (RawData objects)

## Frame Data Structure

### Training Frames (RawData)
Each training frame contains full signal analysis:
- `raw_samples`: Original audio samples (up to 320 samples for 20ms)
- `autocorrelation_vector`: 13 autocorrelation coefficients
- `lpc_vector`: 13 LPC coefficients
- `lsf_vector`: 12 LSF coefficients
- `frame_number`: Frame index within recording
- `recording`: Source recording name
- `parent_centroid_id`: Assigned centroid (after clustering)
- `generation`: LBG generation number

### HMM/Test Frames (RawDataTraining)
Each HMM and test frame contains simplified analysis:
- `raw_samples`: Original audio samples (up to 320 samples for 20ms)
- `autocorrelation_vector`: 13 autocorrelation coefficients
- `frame_number`: Frame index within recording
- `recording`: Source recording name

## Benefits of This Approach

1. **Individual Recording Tracking**: Each recording maintains its identity
2. **Flexible Data Usage**: Can train models on specific recordings or combinations
3. **Reproducible Splits**: Consistent train/test/HMM splits per recording
4