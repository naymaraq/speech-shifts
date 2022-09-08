import torch
from speech_shifts.common.audio.segments import AudioSegment

class WaveformFeaturizer:
    def __init__(self, sample_rate=16000, int_values=False, augmentor=None):
        self._augmentor = augmentor
        self.sample_rate = sample_rate
        self.int_values = int_values

    @property
    def augmentor(self):
        return self._augmentor
    
    def process(self, file_path, offset=0, duration=0, trim=False, orig_sr=None):
        audio = AudioSegment.from_file(
            file_path,
            target_sr=self.sample_rate,
            int_values=self.int_values,
            offset=offset,
            duration=duration,
            trim=trim,
            orig_sr=orig_sr,
        )
        return self.process_segment(audio)

    def process_segment(self, audio_segment):
        if self.augmentor:
            self.augmentor(audio_segment)
        return torch.tensor(audio_segment.samples, dtype=torch.float)