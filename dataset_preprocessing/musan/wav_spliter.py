import copy
import math
import os
import progressbar
import wave
from multiprocessing import Pool
import numpy as np

CHANNELS = 1

def atomic_split(sample):
    orig_audio_path, segments = sample
    orig_audio = wave.open(orig_audio_path, "r")
    items = []
    for segment in segments:

        start = segment["start"]
        end = segment["end"]
        new_wav_filename = segment["wav_filename"]
        file_size = -1
        try:
            split_wav(orig_audio, start, end, new_wav_filename)
            new_wav_filename = os.path.abspath(new_wav_filename)
            if os.path.exists(new_wav_filename):
                file_size = os.path.getsize(new_wav_filename)
        except Exception as excp:
            print(f"Exception accured at atomic_split: {excp}")
        if file_size != -1:
            item = copy.deepcopy(segment)
            item["wav_filename"] = new_wav_filename
            item["wav_filesize"] = file_size
            item["orig_audio_path"] = orig_audio_path
            items.append(item)
    return items


def split_wav(orig_udio, start_time, stop_time, new_wav_file):
    """
    Args:
        orig_udio: returned data from wave.open()
        start_time: float in seconds
        stop_time: float in seconds
        new_wav_file: path to store new wav file
    Returns:
        If there is not extisting file with new_wav_file, create the new file
    """
    if not os.path.exists(new_wav_file):
        frame_rate = orig_udio.getframerate()
        orig_udio.setpos(math.floor(start_time * frame_rate))
        chunk_data = orig_udio.readframes(
            math.ceil((stop_time - start_time) * frame_rate))
        
        audio_as_np_int16 = np.frombuffer(chunk_data, dtype=np.int16)
        if not np.any(audio_as_np_int16):
            print("All entries are zeros -> Skipping {new_wav_file}")
        chunk_audio = wave.open(new_wav_file, "w")
        chunk_audio.setnchannels(CHANNELS)
        chunk_audio.setsampwidth(orig_udio.getsampwidth())
        chunk_audio.setframerate(frame_rate)
        chunk_audio.writeframes(chunk_data)
        chunk_audio.close()


class WavSpliter:
    """Split mono audio files"""

    @staticmethod
    def parallel_split(args: list):
        r"""
        Args:
            args(list of tuples):
                tuple[0](string): "original_audio_path"
                tuple[1](list of dicts):
                    Each dict contains:
                        - "start" (required)
                        - "end" (required)
                        - "wav_filename" (required)
                        - other staff (optional)
        Returns:
            Returns splited audio informations
        """
        pool = Pool()
        items = []
        pbar = progressbar.ProgressBar(max_value=len(args))
        for i, processed in enumerate(pool.imap_unordered(atomic_split, args)):
            items += processed
            pbar.update(i)

        pbar.update(len(args))
        pool.close()
        pool.join()

        return items