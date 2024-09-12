import numpy as np
import sounddevice as sd
import soundfile as sf
from loguru import logger

from src.constants import OUTPUT_FILE_NAME, RECORD_SEC, SAMPLE_RATE


def record_batch(record_sec: int = RECORD_SEC) -> np.ndarray:
    """
    Records an audio batch for a specified duration.

    Args:
        record_sec (int): The duration of the recording in seconds. Defaults to the value of RECORD_SEC.

    Returns:
        np.ndarray: The recorded audio sample.
    """
    logger.debug(f"Recording for {record_sec} second(s)...")

    # Get the default input device (should work with your MacBook Air Microphone)
    device_info = sd.query_devices(kind='input')
    channels = device_info['max_input_channels']
    
    if channels == 0:
        logger.error("No available input channels. Please check your microphone settings.")
        return np.array([])

    try:
        # Record using sounddevice
        audio_sample = sd.rec(int(record_sec * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=channels)
        sd.wait()  # Wait until the recording is finished
        logger.debug("Recording complete.")
        return audio_sample
    except Exception as e:
        logger.error(f"Recording failed: {e}")
        return np.array([])  # Return an empty array on failure


def save_audio_file(audio_data: np.ndarray, output_file_name: str = OUTPUT_FILE_NAME) -> None:
    """
    Saves an audio data array to a file.

    Args:
        audio_data (np.ndarray): The audio data to be saved.
        output_file_name (str): The name of the output file. Defaults to the value of OUTPUT_FILE_NAME.

    Returns:
        None

    Example:
        ```python
        audio_data = np.array([0.1, 0.2, 0.3])
        save_audio_file(audio_data, "output.wav")
        ```
    """
    logger.debug(f"Saving audio file to {output_file_name}...")
    sf.write(file=output_file_name, data=audio_data, samplerate=SAMPLE_RATE)
