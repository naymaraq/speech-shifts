import json
from os.path import expanduser
from typing import Any, Callable, Dict, Iterator, List, Optional, Union


def __parse_item(line: str, manifest_file: str) -> Dict[str, Any]:
    item = json.loads(line)

    # Audio file
    if 'audio_filename' in item:
        item['audio_file'] = item.pop('audio_filename')
    elif 'audio_filepath' in item:
        item['audio_file'] = item.pop('audio_filepath')
    else:
        raise ValueError(
            f"Manifest file {manifest_file} has invalid json line structure: {line} without proper audio file key."
        )
    item['audio_file'] = expanduser(item['audio_file'])

    # Duration.
    if 'duration' not in item:
        raise ValueError(
            f"Manifest file {manifest_file} has invalid json line structure: {line} without proper duration key."
        )

    # Duration.
    if 'label' not in item:
        raise ValueError(
                f"Manifest file {manifest_file} has invalid json line structure: {line} without proper label key."
            )

    item = dict(
        audio_file=item['audio_file'],
        duration=item['duration'],
        offset=item.get('offset', None),
        label=item.get('label', None),
        orig_sr=item.get('orig_sample_rate', None),
    )

    return item


def parse_noise_item(line: str, manifest_file: str) -> Dict[str, Any]:

    item = json.loads(line)

    # Audio file
    if 'audio_filename' in item:
        item['audio_file'] = item.pop('audio_filename')
    elif 'audio_filepath' in item:
        item['audio_file'] = item.pop('audio_filepath')
    else:
        raise ValueError(
            f"Manifest file {manifest_file} has invalid json line structure: {line} without proper audio file key."
        )
    item['audio_file'] = expanduser(item['audio_file'])

    # Duration.
    if 'duration' not in item:
        raise ValueError(
            f"Manifest file {manifest_file} has invalid json line structure: {line} without proper duration key."
        )

    item = dict(
        audio_file=item['audio_file'],
        duration=item['duration'],
        offset=item.get('offset', None),
        label=item.get('label', None),
        orig_sr=item.get('orig_sample_rate', None),
    )

    return item



def item_iter(
        manifests_files: Union[str, List[str]],
        parse_func: Callable[[str, Optional[str]],
                             Dict[str, Any]] = None
) -> Iterator[Dict[str, Any]]:
    """Iterate through json lines of provided manifests.
    NeMo ASR pipelines often assume certain manifest files structure. In
    particular, each manifest file should consist of line-per-sample files with
    each line being correct json dict. Each such json dict should have a field
    for audio file string, a field for duration float and.
    Offset also could be additional field and is set to None by
    default.
    Args:
        manifests_files: Either single string file or list of such -
            manifests to yield items from.
        parse_func: A callable function which accepts as input a single line
            of a manifest and optionally the manifest file itself,
            and parses it, returning a dictionary mapping from str -> Any.
    Yields:
        Parsed key to value item dicts.
    Raises:
        ValueError: If met invalid json line structure.
    """

    if isinstance(manifests_files, str):
        manifests_files = [manifests_files]

    if parse_func is None:
        parse_func = __parse_item

    k = -1
    for manifest_file in manifests_files:
        with open(expanduser(manifest_file), 'r') as f:
            for line in f:
                k += 1
                item = parse_func(line, manifest_file)
                item['id'] = k
                yield item