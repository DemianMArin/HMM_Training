import wave
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import librosa
import soundfile as sf


def display_graphs(x_space: np.ndarray, y_space1: np.ndarray, y_space2: np.ndarray, y_space3: np.ndarray, y_space4: np.ndarray, idx: list):
    x_space = x_space.reshape(-1,1)
    y_space1 = y_space1.reshape(-1,1)
    y_space2 = y_space2.reshape(-1,1)
    y_space3 = y_space3.reshape(-1,1)
    y_space4 = y_space4.reshape(-1,1)


    print(f"{x_space.shape} {y_space1.shape} {y_space2.shape} {y_space3.shape} {y_space4.shape}")
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 12), sharex=True)

    ax1.plot(x_space, y_space1)
    ax1.set_title('Original')
    ax1.set_ylabel('Voltage')

    ax2.plot(x_space, y_space2)
    ax2.set_title('Filtered')
    ax2.set_ylabel('Voltage')

    ax3.plot(x_space, y_space3)
    ax3.set_title('Hamming')
    ax3.set_ylabel('Voltage')

    ax4.plot(x_space, y_space3)
    ax4.axvline(x=x_space[idx[0]], color='r', linestyle='--')
    ax4.axvline(x=x_space[idx[1]], color='r', linestyle='--')
    ax4.set_title('Power & ZCR')
    ax4.set_ylabel('Voltage')

    ax5.plot(x_space[idx[0]:idx[1]], y_space4)
    ax5.set_title('Trimmed')
    ax5.set_ylabel('Voltage')


    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)

    plt.show()

def save2file(array_to_write: np.ndarray, name_file: str = "demofile.txt"):
    with open(name_file, "w") as f:
        count = len(array_to_write) 
        # count = 10
        for i in range(array_to_write.shape[0]):
            if i < count:
                # print(f"{array_to_write[i][0]}")
                f.write(f"{array_to_write[i][0]}\n")
            if i > count: break
# INFO: Check and convert audio metadata to 16kHz mono if needed
# Input: path of .wav file
# Output: converted file path (same path if no conversion needed)
def check_metadata(file_path: str, print_messages=False):
    # Load audio file with librosa to check metadata
    y, sr = librosa.load(file_path, sr=None, mono=False)
    
    # Check if conversion is needed
    needs_conversion = False
    conversion_reasons = []
    
    # Check sample rate
    if sr != 16000:
        needs_conversion = True
        conversion_reasons.append(f"sample rate {sr}Hz -> 16000Hz")
    
    # Check if stereo (2 channels) - librosa.load with mono=False preserves original channels  
    if len(y.shape) > 1:
        needs_conversion = True
        conversion_reasons.append(f"channels {y.shape[0]} -> 1 (mono)")
    
    # Print conversion message if needed
    if needs_conversion and print_messages:
        print(f"Converting {file_path}: {', '.join(conversion_reasons)}")
    
    # Convert if needed
    if needs_conversion:
        # Convert to mono and resample to 16kHz
        y_mono = librosa.to_mono(y) if len(y.shape) > 1 else y
        y_resampled = librosa.resample(y_mono, orig_sr=sr, target_sr=16000)
        
        # Save the converted file (overwrite original)
        sf.write(file_path, y_resampled, 16000)
        
        return file_path
    
    return file_path

#INFO: Opening file. 
# Input: path of .wav file
# Output: frames array, nframes int 
def open_file(file_with_signal: str, print_path=False):
    with wave.open(f"{file_with_signal}.wav") as wav_file:
        metadata = wav_file.getparams() 
        frames = wav_file.readframes(metadata.nframes)
        sampwidth = wav_file.getsampwidth()
        nchannels = wav_file.getnchannels()
        nframes = wav_file.getnframes()
        framerate = wav_file.getframerate()

    if (print_path):
        print(f"F() -> open_file")
        print(f"Path: {file_with_signal}")
        # print(f"Type frames: {type(frames)}")
        print(f"Metadata: {metadata} ")

    signal_data = {
        "metadata" : metadata,
        "frames" : frames,
        "sampwidth" : sampwidth,
        "nchannels" : nchannels,
        "nframes" : nframes,
        "framerate" : framerate 
    }

    return signal_data

# INFO: Assigning a step time of 1/framerate to each samples_with_time.
# "nframes" are the number of samples (each point in the signal recorded)
# Decoding the raw "frames" from wav_file
# Output: time array, array_sample array 
def assing_step_time(frames: bytes, framerate: int, nchannels: int, nframes: int):
    array_sample = np.frombuffer(frames,dtype=np.int16).reshape(-1,nchannels)
    step = 1/framerate
    time = np.arange(0, nframes*step, step).reshape(-1,1)
    # samples_with_time = np.hstack((array_sample, time)) 

    # print(f"Len {len(samples_with_time)}. Shape: {samples_with_time.shape}")

    output = {
        "array_sample" : array_sample,
        "time" : time
        }

    return output

# INFO: Applying `1-(0.95z^-1)` filter
# Output: array_sample array
def filter_signal(signal: np.ndarray, nframes: int):
    top = 10
    filtered_signal = np.zeros((nframes, 1))
    for i in range(nframes-1):
        # print(f"{filtered_signal[i+1, 0]} = {array_sample[i+1, 0]} - {0.95*array_sample[i,0]}")
        filtered_signal[i+1, 0] = signal[i+1, 0] - 0.95*signal[i, 0]
        # if i > top: break

    output = {"filtered_signal" : filtered_signal}
    return output


# INFO: Applying Hamming window
# NOTE: In this case hamming window and frame are the same length.
# Output: hamming_signal array
def hamming_window(signal: np.ndarray, nframes: int):
    samples_per_hamming = 320
    samples_per_hop = 128
    num_frames_in_signal_hamming = int((nframes-samples_per_hamming)/samples_per_hop) + 1

    indices_array = np.arange(samples_per_hamming)
    hamming_window = 0.54 - 0.46 * np.cos((2 * np.pi * indices_array)/ (samples_per_hamming-1))

    top = nframes
    bottom = 10
    hamming_signal = np.copy(signal)
    # print(f"Num frames in signal hamming: {num_frames_in_signal_hamming}")
    for i in range(num_frames_in_signal_hamming+1):
        start = i*samples_per_hop
        finish = start + samples_per_hamming

        if i == num_frames_in_signal_hamming:  
            last_samples = len(hamming_signal[start:-1,0])
            hamming_signal[start:-1,0] = hamming_signal[start:-1,0] * hamming_window[0:last_samples]
        else:
            hamming_signal[start:finish,0] = hamming_signal[start:finish,0] * hamming_window 

    output = {"hamming_signal" : hamming_signal}
    return output


# INFO: Calculate Start and Finish with Energy and ZCR
# and get trimmed signal. It removes empty space.
# We must divide into 20ms frames with 10ms overlap
# framerate => 1s
# x         => 20ms
# x -> samples per frame
# Outuput: sliced array 
def slice_signal(signal: np.ndarray, framerate: int, nframes: int):
    samples_per_20ms = int(0.02 * framerate)
    samples_per_10ms = int(0.01 * framerate)
    num_frames_in_signal = int((nframes-samples_per_20ms)/samples_per_10ms) + 1

    zcr = np.zeros((num_frames_in_signal,1))
    power = np.zeros((num_frames_in_signal,1))

    top = 13
    bottom = 10
    # print(f"Num frames in signal: {num_frames_in_signal}")
    for i in range(num_frames_in_signal):
        start = i*samples_per_10ms
        finish = start + samples_per_20ms

        if i == num_frames_in_signal-1:  
            frame_to_evalute = signal[start:-1].reshape(-1,1)
        else:
            frame_to_evalute = signal[start:finish,0].reshape(-1,1)

        # Calculating ZCR = 1/2 * Sum(sign(n) - sing(n-1))
        frame_to_evalute_shifted = np.roll(frame_to_evalute, -1)
        frame_to_evalute_shifted[-1,0] = 0

        differences = np.subtract(np.sign(frame_to_evalute_shifted), np.sign(frame_to_evalute))
        differences[-1,0] = 0
        abs_array = np.abs(differences)
        sum = np.sum(abs_array)
        zcr[i,0] = sum / 2;

        # Calculating Power = Sum(x^2) / x_len
        power[i,0] = np.sum(frame_to_evalute**2)/len(frame_to_evalute)

    # max_zcr = 0.08*np.max(zcr) # Thresholds
    # max_power = 0.03*np.max(power)
    max_zcr = 0.001*np.max(zcr) # Thresholds
    max_power = 0.001*np.max(power)


    zcr_upper_array = np.where(zcr>max_zcr, 1, 0)
    power_upper_array = np.where(power>max_power, 1, 0)

    # If the frame has zcr and power greater than the Thresholds
    # then the frame will stay
    zcr_and_power_upper = np.logical_and(zcr_upper_array, power_upper_array).astype(int)

    if np.any(zcr_and_power_upper==1):
        first = np.where(zcr_and_power_upper[:,0]==1)[0][0]
        last = np.where(zcr_and_power_upper[:,0]==1)[0][-1]
    else:
        print("No 1 founded in ZCR and Power")
        first = 0
        last = len(zcr_upper_array)

    start_index_trimmed_sampled = first*samples_per_10ms
    finish_index_trimmed_sampled = last*samples_per_10ms

    trimmed_signal = signal[start_index_trimmed_sampled:finish_index_trimmed_sampled]
    
    # with wave.open("output.wav", mode="wb") as wav_file:
    #     wav_file.setnchannels(nchannels)
    #     wav_file.setsampwidth(sampwidth)
    #     wav_file.setframerate(framerate)
    #     wav_file.writeframes(trimmed_signal.tobytes())

    output = {
        "trimmed_signal" : trimmed_signal,
        "start_idx" : start_index_trimmed_sampled,
        "finish_idx" : finish_index_trimmed_sampled
      }
    return output

def do_preemphasis(path: str, output_path: str, display_graphs_allow=False, save_to_file=False, print_messages = False):
    # Check and convert metadata if needed
    check_metadata(path + ".wav", print_messages=print_messages)
    
    signal_data = open_file(path, print_path=False)
    frames = signal_data["frames"]
    nchannels = signal_data["nchannels"]
    nframes = signal_data["nframes"]
    framerate = signal_data["framerate"]

    output = assing_step_time(frames, framerate, nchannels, nframes)
    signal_raw = output["array_sample"]
    time = output["time"]

    output = filter_signal(signal_raw, nframes)
    signal_filtered = output["filtered_signal"]

    output = hamming_window(signal_filtered, nframes)
    signal_hamming = output["hamming_signal"]

    output = slice_signal(signal_hamming, framerate, nframes)
    trimmed_signal = output["trimmed_signal"]
    start_idx = output["start_idx"]
    finish_idx = output["finish_idx"]


    if (display_graphs_allow == True):
        print(f"Trimmed signal shape: {trimmed_signal.shape}")
        display_graphs(time, signal_raw, signal_filtered, signal_hamming, trimmed_signal, [start_idx, finish_idx])

    if (save_to_file == True):
        np.save(output_path+".npy", trimmed_signal)


if __name__ == "__main__":
    print(f"Starting preemphasis")

    raw_data_path = "Data/Raw/"
    processed_data_path = "Data/Processed/"
    
    # Get all folders in Data/Raw/
    folders = [f for f in os.listdir(raw_data_path) if os.path.isdir(os.path.join(raw_data_path, f))]
    
    for folder in folders:
        input_folder_path = os.path.join(raw_data_path, folder)
        output_folder_path = os.path.join(processed_data_path, folder)
        
        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
        
        # Get all .wav files in the current folder
        wav_files = [f for f in os.listdir(input_folder_path) if f.endswith('.wav')]
        
        for wav_file in tqdm(wav_files, desc=f"Processing {folder}"):
            # Remove .wav extension to get base name
            base_name = wav_file[:-4]
            
            input_file_path = os.path.join(input_folder_path, base_name)
            output_file_path = os.path.join(output_folder_path, base_name)
            
            do_preemphasis(input_file_path, output_file_path, display_graphs_allow=False, save_to_file=True, print_messages=True)
