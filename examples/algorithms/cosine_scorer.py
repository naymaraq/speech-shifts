from torchmetrics import Metric
import torch

def item(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.item()
    return tensor

class CosineScorer(Metric):
    def __init__(self, 
                    input_trial_array, 
                    index2path,
                    mean_norm=False,
                    std_norm=False, 
                    score_fn=None):
        super().__init__(dist_sync_on_step=True)
        self.input_trial_array = input_trial_array
        self.index2path = index2path
        self.mean_norm = mean_norm
        self.std_norm = std_norm
        self.score_fn = score_fn if score_fn is not None else lambda x: (x+1)/2

        self.add_state("embeddings", default=[], persistent=False)
        self.add_state("indices", default=[], persistent=False)
        self.eps = 1e-10
        self.disable_compute()

    def enable_compute(self):
        self.run_eer = True
    
    def disable_compute(self):
        self.run_eer = False
    
    def get_embeddings(self, to_numpy=False):
        all_embs = torch.vstack(self.embeddings)
        all_ids  = torch.hstack(self.indices)
        all_embs = self.normalize_embeddings(all_embs)
        out_embeddings = {}    
        for i, audio_file_id in enumerate(all_ids):
            audio_file_id = item(audio_file_id)
            audio_path = self.index2path[audio_file_id]
            if to_numpy:
                out_embeddings[audio_path] = all_embs[i].detach().cpu().numpy()
            else:
                out_embeddings[audio_path] = all_embs[i]
        return out_embeddings
            
    def normalize_embeddings(self, embs):
        if self.mean_norm:
            embs = embs - torch.mean(embs, axis=0)
        if self.std_norm:
            embs = embs / (embs.std(axis=0) + self.eps)
        embs_l2_norm = torch.linalg.norm(embs, ord=2, dim=-1, keepdim=True)
        embs = embs / embs_l2_norm
        return embs

    def update(self, embeddings, indices):
        assert embeddings.shape[0] == len(indices)
        self.embeddings.append(embeddings)
        self.indices.append(indices)

    def compute(self):
        if self.run_eer:
            out_embeddings = self.get_embeddings(to_numpy=False)
            
            y_pred, missed_trials = [], 0
            y_true = []
            for y, x_speaker, y_speaker in self.input_trial_array:
                if x_speaker in out_embeddings and y_speaker in out_embeddings:
                    X = out_embeddings[x_speaker]
                    Y = out_embeddings[y_speaker]
                    score = torch.dot(X, Y) / ((torch.dot(X, X) * torch.dot(Y, Y)) ** 0.5)
                    score = self.score_fn(score)
                    y_pred.append(item(score))
                    y_true.append(item(y))
                else:
                    missed_trials += 1
            
            if missed_trials > 0:
                print(f"Missed trials detected: {missed_trials}")
            
            y_pred = torch.tensor(y_pred).float()
            y_true = torch.tensor(y_true).float()
            return y_pred, y_true

        