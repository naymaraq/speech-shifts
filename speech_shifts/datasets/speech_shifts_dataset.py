import torch
import numpy as np

class SpeechShiftsDataset:
    DEFAULT_SPLITS = {'train': 0, 'val': 1, 'test': 2}
    DEFAULT_SPLIT_NAMES = {'train': 'Train', 'val': 'Validation', 'test': 'Test'}
    
    @property
    def n_classes(self):
        """
        Number of classes for single-task classification datasets.
        Used for logging and to configure models to produce appropriately-sized output.
        None by default.
        Leave as None if not applicable (e.g., regression or multi-task classification).
        """
        return getattr(self, '_n_classes', None)
    
    @property
    def y_array(self):
        """
        A Tensor of targets (e.g., labels for classification tasks),
        with y_array[i] representing the target of the i-th data point.
        y_array[i] can contain multiple elements.
        """
        return self._y_array

    @property
    def y_size(self):
        """
        The number of dimensions/elements in the target, i.e., len(y_array[i]).
        For standard classification/regression tasks, y_size = 1.
        For multi-task or structured prediction settings, y_size > 1.
        Used for logging and to configure models to produce appropriately-sized output.
        """
        return self._y_size

    @property
    def dataset_name(self):
        """
        A string that identifies the dataset.
        """
        return self._dataset_name
    
    @property
    def metadata_fields(self):
        """
        A list of strings naming each column of the metadata table.
        Must include 'y'.
        """
        return self._metadata_fields
    
    @property
    def metadata_array(self):
        """
        A Tensor of metadata, with the i-th row representing the metadata associated with
        the i-th data point. The columns correspond to the metadata_fields defined above.
        """
        return self._metadata_array

    @property
    def metadata_map(self):
        """
        An optional dictionary that, for each metadata field, contains a list that maps from
        integers (in metadata_array) to a string representing what that integer means.
        This is only used for logging, so that we print out more intelligible metadata values.
        Each key must be in metadata_fields.
        """
        return getattr(self, '_metadata_map', None)
    
    @property
    def split_dict(self):
        """
        A dictionary mapping splits to integer identifiers (used in split_array),
        e.g., {'train': 0, 'val': 1, 'test': 2}.
        Keys should match up with split_names.
        """
        return getattr(self, '_split_dict', SpeechShiftsDataset.DEFAULT_SPLITS)

    @property
    def split_names(self):
        """
        A dictionary mapping splits to their pretty names,
        e.g., {'train': 'Train', 'val': 'Validation', 'test': 'Test'}.
        Keys should match up with split_dict.
        """
        return getattr(self, '_split_names', SpeechShiftsDataset.DEFAULT_SPLIT_NAMES)
    
    @property
    def split_array(self):
        """
        An array of integers, with split_array[i] representing what split the i-th data point
        belongs to.
        """
        return self._split_array
    
    @property
    def duration_array(self):
        """
        An array of floats, with duration_array[i] representing duration ofthe i-th data point.
        """
        return self._duration_array
    
    @property
    def collate(self):
        """
        Torch function to collate items in a batch.
        By default returns None -> uses default torch collate.
        """
        return getattr(self, '_collate', None)
     
    def get_input(self, idx):
        """
        Args:
            - idx (int): Index of a data point
        Output:
            - x (Tensor): Input features of the idx-th data point
        """
        raise NotImplementedError
    
    def get_subset(self, split, loader_kwargs={"type": "single_view"}, frac=1.0, min_dur=None, max_dur=None):
        """
        Args:
            - split (str): Split identifier, e.g., 'train', 'val', 'test'.
                           Must be in self.split_dict.
            - frac (float): What fraction of the split to randomly sample.
                            Used for fast development on a small dataset.
            - transform (function): Any data transformations to be applied to the input x.
        Output:
            - subset (SpeechShiftsSubset): A (potentially subsampled) subset of the SpeechShiftsDataset.
        """
        if split not in self.split_dict:
            raise ValueError(f"Split {split} not found in dataset's split_dict.")

        split_mask = self.split_array == self.split_dict[split]
        split_idx = np.where(split_mask)[0]

        if frac < 1.0:
            # Randomly sample a fraction of the split
            num_to_retain = int(np.round(float(len(split_idx)) * frac))
            split_idx = np.sort(np.random.permutation(split_idx)[:num_to_retain])
        
        if min_dur:
            min_duration_mask = self.duration_array >= min_dur
            min_dur_idx = np.where(min_duration_mask)[0]
            split_idx = np.intersect1d(split_idx, min_dur_idx)
        
        if max_dur:
            max_duration_mask = self.duration_array <= max_dur
            max_dur_idx = np.where(max_duration_mask)[0]
            split_idx = np.intersect1d(split_idx, max_dur_idx)

        return SpeechShiftsSubset(self, split_idx, loader_kwargs)

    def get_augmented_item(self, idx, augmentor):
        x = self.get_input(idx, augmentor=augmentor)
        y = self.y_array[idx]
        metadata = self.metadata_array[idx]
        return x, y, metadata

    def __getitem__(self, idx):
        x = self.get_input(idx)
        y = self.y_array[idx]
        metadata = self.metadata_array[idx]
        return x, y, metadata

    def __len__(self):
        return len(self.y_array)

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.check_init()
    
    def check_init(self):
        required_attrs = ['_dataset_name',
                          '_split_array',
                          '_y_array', '_y_size', '_duration_array',
                          '_metadata_fields', '_metadata_array']
        for attr_name in required_attrs:
            assert hasattr(self, attr_name), f'{attr_name} is missing.'


        # Check splits
        assert self.split_dict.keys()==self.split_names.keys()
        assert 'train' in self.split_dict
        assert 'val' in self.split_dict

        # Check the form of the required arrays
        assert (isinstance(self.y_array, torch.Tensor) or isinstance(self.y_array, list))
        assert isinstance(self.metadata_array, torch.Tensor), 'metadata_array must be a torch.Tensor'

        # Check that dimensions match
        assert len(self.y_array) == len(self.metadata_array)
        assert len(self.split_array) == len(self.metadata_array)
        assert len(self.duration_array) == len(self.metadata_array)

        # Check metadata
        assert len(self.metadata_array.shape) == 2
        assert len(self.metadata_fields) == self.metadata_array.shape[1]

        # For convenience, include y in metadata_fields if y_size == 1
        if self.y_size == 1:
            assert 'y' in self.metadata_fields
        
    @staticmethod
    def standard_eval(metric, y_pred, y_true):
        """
        Args:
            - metric (Metric): Metric to use for eval
            - y_pred (Tensor): Predicted targets
            - y_true (Tensor): True targets
        Output:
            - results (dict): Dictionary of results
            - results_str (str): Pretty print version of the results
        """
        results = metric.compute(y_pred, y_true, return_dict=True)
        results_str = (
            f"Average {metric.name}: {results[metric.agg_metric_field]:.3f}\n"
        )
        return results, results_str

    @staticmethod
    def standard_group_eval(metric, grouper, y_pred, y_true, metadata, aggregate=True):
        """
        Args:
            - metric (Metric): Metric to use for eval
            - grouper (CombinatorialGrouper): Grouper object that converts metadata into groups
            - y_pred (Tensor): Predicted targets
            - y_true (Tensor): True targets
            - metadata (Tensor): Metadata
        Output:
            - results (dict): Dictionary of results
            - results_str (str): Pretty print version of the results
        """
        results, results_str = {}, ''
        if aggregate:
            results.update(metric.compute(y_pred, y_true))
            results_str += f"Average {metric.name}: {results[metric.agg_metric_field]:.3f}\n"
        
        g = grouper.metadata_to_group(metadata)
        group_results = metric.compute_group_wise(y_pred, y_true, g, grouper.n_groups)
        for group_idx in range(grouper.n_groups):
            group_str = grouper.group_field_str(group_idx)
            group_metric = group_results[metric.group_metric_field(group_idx)]
            group_counts = group_results[metric.group_count_field(group_idx)]
            results[f'{metric.name}_{group_str}'] = group_metric
            results[f'count_{group_str}'] = group_counts
            if group_results[metric.group_count_field(group_idx)] == 0:
                continue
            results_str += (
                f'  {grouper.group_str(group_idx)}  '
                f"[n = {group_results[metric.group_count_field(group_idx)]:6.0f}]:\t"
                f"{metric.name} = {group_results[metric.group_metric_field(group_idx)]:5.3f}\n")
        results[f'{metric.worst_group_metric_field}'] = group_results[f'{metric.worst_group_metric_field}']
        results_str += f"Worst-group {metric.name}: {group_results[metric.worst_group_metric_field]:.3f}\n"
        return results, results_str


class SpeechShiftsSubset(SpeechShiftsDataset):

    def __init__(self, dataset, indices, loader_kwargs, augmentor=None):

        self.dataset = dataset
        self.augmentor = augmentor
        self.indices = indices
        inherited_attrs = ['_dataset_name',
                           '_split_dict', '_split_names',
                           '_y_size', '_n_classes',
                           '_metadata_fields', 
                           '_metadata_map', '_index2path']
        
        for attr_name in inherited_attrs:
            if hasattr(dataset, attr_name):
                setattr(self, attr_name, getattr(dataset, attr_name))
        
        self.loader_kwargs = loader_kwargs
        self.n_views = loader_kwargs["n_views"]
        if self.n_views == 1:
            self._collate = self.dataset._collate
        elif self.n_views > 1:
            self.label2samples = {}
            for idx in range(len(self.indices)):
                y = self.dataset.y_array[self.indices[idx]].item()
                if y not in self.label2samples:
                    self.label2samples[y] = []
                self.label2samples[y].append(idx)
            self._collate = _multivew_collate_fn
        else:
            raise ValueError("n_views need to be greater than 1")
    
    def standard_getitem(self, idx):
        x, y, metadata = self.dataset.get_augmented_item(self.indices[idx], self.augmentor)
        return x, y, metadata

    def get_sample(self, idx):
        y = self.dataset.y_array[self.indices[idx]]
        anchor_id = np.random.choice(self.label2samples[y.item()])
        x, y, metadata = self.standard_getitem(anchor_id)
        return x, y, metadata

    def multiview_getitem(self, idx):
        views = []
        for _ in range(self.n_views):
            view = self.get_sample(idx)
            views.append(view)
        return views

    def __getitem__(self, idx):
        if self.n_views == 1:
            return self.standard_getitem(idx)
        else:
            return self.multiview_getitem(idx)

    def __len__(self):
        return len(self.indices)
    
    @property
    def split_array(self):
        return self.dataset._split_array[self.indices]

    @property
    def y_array(self):
        return self.dataset._y_array[self.indices]

    @property
    def metadata_array(self):
        return self.dataset.metadata_array[self.indices]
    
    @property
    def duration_array(self):
        return self.dataset.duration_array[self.indices]

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        return self.dataset.eval(y_pred, y_true, metadata, prediction_fn)


class SpeechShiftsSubsetWithTrials(SpeechShiftsSubset):

    def __init__(self, dataset, indices, trial_indices, loader_kwargs, augmentor=None):

        self.trial_indices = trial_indices
        inherited_trial_attrs = ['_trial_y_size', 
                                 '_trial_n_classes',
                                 '_trial_metadata_fields', 
                                 '_trial_metadata_map']

        for attr_name in inherited_trial_attrs:
            if hasattr(dataset, attr_name):
                setattr(self, attr_name, getattr(dataset, attr_name))
        
        super().__init__(dataset, indices, loader_kwargs, augmentor)
    
    @property
    def trial_split_array(self):
        return self.dataset._trial_split_array[self.trial_indices]

    @property
    def trial_y_array(self):
        return self.dataset._trial_y_array[self.trial_indices]

    @property
    def trial_metadata_array(self):
        return self.dataset._trial_metadata_array[self.trial_indices]

    @property
    def input_trial_array(self):
        return [self.dataset._input_trial_array[ix] for ix in self.trial_indices]

def _multivew_collate_fn(batch):
    views = []
    for i in range(len(batch)):
        views.extend(batch[i])

    return _fixed_seq_collate_fn(views)

def _fixed_seq_collate_fn(batch):
    """collate batch of audio sig, audio len, label, metadata"""
    sig_and_length, _, _ = zip(*batch)
    audio_lengths = [length for _, length, _ in sig_and_length]
    fixed_length = int(max(audio_lengths))

    indices = []
    audio_signal, labels, new_audio_lengths, metadata = [], [], [], []
    for (sig, sig_len, idx), labels_i, meta in batch:
        sig_len = sig_len.item()
        if sig_len < fixed_length:
            repeat = fixed_length // sig_len
            rem = fixed_length % sig_len
            sub = sig[-rem:] if rem > 0 else torch.tensor([])
            rep_sig = torch.cat(repeat * [sig])
            sig = torch.cat((rep_sig, sub))
        
        new_audio_lengths.append(torch.tensor(fixed_length))    
        audio_signal.append(sig)
        labels.append(labels_i)
        metadata.append(meta)
        indices.append(torch.tensor(idx).long())

    audio_signal = torch.stack(audio_signal)
    audio_lengths = torch.stack(new_audio_lengths)
    labels = torch.stack(labels)
    metadata = torch.stack(metadata)
    indices = torch.stack(indices)
    return audio_signal, audio_lengths, labels, metadata, indices
    
    

