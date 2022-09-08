import collections
import os
from typing import List, Optional, Union

from examples.augmentations.manifest import item_iter


class _Collection(collections.UserList):
    """List of parsed and preprocessed data."""

    OUTPUT_TYPE = None  # Single element output type.


class SpeechLabel(_Collection):
    """List of audio-transcript text correspondence with preprocessing."""

    OUTPUT_TYPE = collections.namedtuple(
        typename='SpeechLabelEntity',
        field_names='audio_file duration label offset',)

    def __init__(
            self,
            audio_files: List[str],
            durations: List[float],
            labels: List[Union[int, str]],
            offsets: List[Optional[float]],
            min_duration: Optional[float] = None,
            max_duration: Optional[float] = None,
            max_number: Optional[int] = None,
            do_sort_by_duration: bool = False,
            index_by_file_id: bool = False,
    ):
        """Instantiates audio-text manifest with filters and preprocessing.
        Args:
            audio_files: List of audio files.
            durations: List of float durations.
            offsets: List of duration offsets or None.
            min_duration: Minimum duration to keep entry with (default: None).
            max_duration: Maximum duration to keep entry with (default: None).
            max_number: Maximum number of samples to collect.
            do_sort_by_duration: True if sort samples list by duration. Not compatible with index_by_file_id.
            index_by_file_id: If True, saves a mapping from filename base (ID) to index in data.
        """

        if index_by_file_id:
            self.mapping = {}
        output_type = self.OUTPUT_TYPE
        data, duration_filtered = [], 0.0
        for audio_file, duration, label, offset in zip(audio_files, durations, labels, offsets):
            # Duration filters.
            if min_duration is not None and duration < min_duration:
                duration_filtered += duration
                continue

            if max_duration is not None and duration > max_duration:
                duration_filtered += duration
                continue

            data.append(output_type(audio_file, duration, label, offset))

            if index_by_file_id:
                file_id, _ = os.path.splitext(os.path.basename(audio_file))
                self.mapping[file_id] = len(data) - 1

            # Max number of entities filter.
            if len(data) == max_number:
                break

        if do_sort_by_duration:
            if index_by_file_id:
                print("Tried to sort dataset by duration, but cannot since index_by_file_id is set.")
            else:
                data.sort(key=lambda entity: entity.duration)

        self.uniq_labels = sorted(set(map(lambda x: x.label, data)))
        super().__init__(data)


class SpeechLabelManifestProcessor(SpeechLabel):
    """`AudioText` collector from asr structured json files."""

    def __init__(self, manifests_files: Union[str, List[str]],
                 parse_func=None,
                 *args, **kwargs):
        """Parse lists of audio files, durations.
        Args:
            manifests_files: Either single string file or list of such -
                manifests to yield items from.
            *args: Args to pass to `AudioText` constructor.
            **kwargs: Kwargs to pass to `AudioText` constructor.
        """

        audio_files, durations, offsets, labels = [], [], [], []
        for item in item_iter(manifests_files, parse_func=parse_func):
            audio_files.append(item['audio_file'])
            durations.append(item['duration'])
            offsets.append(item['offset'])
            labels.append(item['label'])

        super().__init__(audio_files=audio_files,
                         durations=durations,
                         offsets=offsets,
                         labels=labels,
                         *args,
                         **kwargs)