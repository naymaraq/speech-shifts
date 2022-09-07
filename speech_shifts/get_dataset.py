import speech_shifts

def get_dataset(dataset: str, **dataset_kwargs):

    if dataset not in speech_shifts.supported_datasets:
        raise ValueError(f'The dataset {dataset} is not recognized. Must be one of {speech_shifts.supported_datasets}.')
    
    if dataset == 'MLSR':
        from  speech_shifts.datasets.mlsr_dataset import MLSRDataset
        return MLSRDataset(**dataset_kwargs)