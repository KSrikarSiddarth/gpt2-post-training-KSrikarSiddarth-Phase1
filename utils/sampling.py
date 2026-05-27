from transformers import LogitsProcessor


class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    def __init__(self, penalty: float):
        self.penalty = penalty

    def __call__(self, input_ids, scores):
        for i in range(input_ids.shape[0]):
            for token_id in input_ids[i].unique():
                token_id = token_id.item()
                if scores[i, token_id] < 0:
                    scores[i, token_id] *= self.penalty
                else:
                    scores[i, token_id] /= self.penalty
        return scores


class DistributionRecorder(LogitsProcessor):
    """Captures post-filter logits at each generation step (no-op on scores)."""

    def __init__(self, on_step=None):
        self.records = []   # list of CPU tensors, one per step
        self._on_step = on_step

    def __call__(self, input_ids, scores):
        step = len(self.records)
        self.records.append(scores.detach().cpu().clone())
        if self._on_step is not None:
            self._on_step(step, scores.detach().cpu())
        return scores
