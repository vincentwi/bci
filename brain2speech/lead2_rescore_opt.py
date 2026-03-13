#!/usr/bin/env python3
"""OPT-6.7B N-best rescoring for DCoND decoder.

NOTE: This was planned but NOT used in the final pipeline.
The phoneme-to-text LIFT approach (lead2_train_lift.py) bypasses the need for
phoneme→word conversion + OPT rescoring entirely. LIFT achieves 28.1% val WER
directly from phonemes, which is better than the CMUdict+OPT pipeline would give.

The final Lead 2 pipeline is:
  DCoND (greedy CTC) → LIFT v1 (phoneme→text) → Optional corrector

This script is kept for reference/future work if someone wants to try the
traditional WFST + OPT rescoring pipeline from the DCoND paper.

Pipeline (from arxiv 2411.10657):
1. DCoND → marginalized monophone log probs
2. KenLM beam search → top-100 phoneme hypotheses
3. Phoneme→word conversion via CMUdict reverse lookup
4. OPT-6.7B scores each word-level candidate by log-probability
5. Re-rank by: ctc_score + lm_weight * opt_score
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class OPTRescorer:
    def __init__(self, model_name='facebook/opt-6.7b', device='cuda'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map='auto')
        self.model.eval()

    def score(self, text):
        """Compute sequence log-probability under OPT-6.7B."""
        inputs = self.tokenizer(text, return_tensors='pt').to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            log_probs = outputs.logits.log_softmax(dim=-1)
            token_scores = log_probs[0, :-1].gather(
                1, inputs.input_ids[0, 1:].unsqueeze(1))
            return token_scores.sum().item() / max(1, token_scores.shape[0])

    def rescore_nbest(self, candidates, ctc_scores, lm_weight=0.5):
        """Re-rank N-best list using OPT-6.7B scores."""
        scored = []
        for text, ctc_s in zip(candidates, ctc_scores):
            opt_s = self.score(text)
            scored.append((text, ctc_s + lm_weight * opt_s))
        return sorted(scored, key=lambda x: x[1], reverse=True)


if __name__ == '__main__':
    print("OPT rescoring not used in final pipeline.")
    print("See lead2_train_lift.py for the phoneme→text LIFT approach instead.")
