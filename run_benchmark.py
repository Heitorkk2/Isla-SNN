import os
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

import lm_eval
from lm_eval.api.model import LM

import isla

class IslaWrapper(LM):
    def __init__(self, model, tokenizer, device, batch_size=16):
        """
        Wrapper for Isla-SNN compatible with lm-evaluation-harness.
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self._device = device
        self.batch_size_per_gpu = batch_size

    def _batch_requests(self, requests, n):
        """Helper to yield successive n-sized chunks from requests."""
        for i in range(0, len(requests), n):
            yield requests[i:i + n]

    def loglikelihood(self, requests):
        """
        Calculate log-likelihood of (context, continuation) pairs.
        For exactness and stability across our SNN, we run batch_size=1 sequentially,
        but you can modify this to pad sequences for batch>1.
        """
        results = []
        for req in tqdm(requests, desc="Evaluating LogLikelihood"):
            context, continuation = req.args

            ctx_enc = self.tokenizer(context, return_tensors='pt', add_special_tokens=False).input_ids[0]
            cont_enc = self.tokenizer(continuation, return_tensors='pt', add_special_tokens=False).input_ids[0]

            # full sequence: context + continuation
            full_enc = torch.cat([ctx_enc, cont_enc]).unsqueeze(0).to(self._device)
            ctx_len = ctx_enc.shape[0]
            full_len = full_enc.shape[1]

            with torch.no_grad():
                logits, _, _ = self.model(full_enc) # [1, L, V]

            # We want log probs of the continuation. 
            # Prediction for token i is logits at i-1
            cont_logits = logits[0, ctx_len-1 : full_len-1, :] # [cont_len, V]
            cont_logprobs = F.log_softmax(cont_logits, dim=-1)

            # Get logprobs of the actual continuation tokens
            chosen_logprobs = cont_logprobs.gather(
                dim=-1, 
                index=cont_enc.to(self._device).unsqueeze(-1)
            ).squeeze(-1)

            sum_logprob = chosen_logprobs.sum().item()

            # is_greedy: check if the argmax of logits exactly matches cont_enc
            pred_tokens = cont_logits.argmax(dim=-1)
            is_greedy = (pred_tokens == cont_enc.to(self._device)).all().item()

            # result format: (loglikelihood, is_greedy)
            results.append((sum_logprob, is_greedy))

        return results

    def generate_until(self, requests):
        """
        Generate text until a stop sequence.
        """
        results = []
        for req in tqdm(requests, desc="Generating"):
            context = req.args[0]
            # req.args[1] is a dict with optional keys like 'until', 'max_gen_toks'
            args = req.args[1] if len(req.args) > 1 else {}
            
            until = args.get('until', [self.tokenizer.eos_token])
            if isinstance(until, str):
                until = [until]
                
            max_new = args.get('max_gen_toks', 64)

            gen_text = ""
            for word in isla.generate_stream(
                self.model, 
                self.tokenizer, 
                context, 
                max_new_tokens=max_new, 
                device=self._device
            ):
                gen_text += word
                if any(gen_text.endswith(stop) for stop in until):
                    # remove the stop sequence
                    for stop in until:
                        if gen_text.endswith(stop):
                            gen_text = gen_text[:-len(stop)]
                            break
                    break

            results.append(gen_text)

        return results

    def loglikelihood_rolling(self, requests):
        return []

def run_benchmarks():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Isla-SNN on {device}...")
    model, tokenizer = isla.load_model("outputs/checkpoints/final", device=device)
    model.eval()

    # Create wrapper
    lm_obj = IslaWrapper(model, tokenizer, device=device)

    # Benchmark tasks
    tasks = ["hellaswag", "arc_challenge", "piqa", "winogrande"]
    print(f"Running evaluation on: {tasks}...")

    # lm_eval >= v0.4.0 format
    results = lm_eval.simple_evaluate(
        model=lm_obj,
        tasks=tasks,
        num_fewshot=0,
        batch_size=1
    )

    print("Evaluation Complete!")

    # Parse results
    data = []
    for task, metrics in results['results'].items():
        row = {'Task': task}
        score = None
        priority_keys = ['acc_norm,none', 'acc_norm', 'acc,none', 'acc']

        for key in priority_keys:
            if key in metrics:
                score = metrics[key]
                break

        if score is None:
            for k, v in metrics.items():
                if isinstance(v, float):
                    score = v
                    break

        row['Score'] = score if score is not None else "N/A"
        data.append(row)

    df_results = pd.DataFrame(data)
    print("\n--- Benchmark Results ---")
    print(df_results.to_markdown(index=False))

    # Save to CSV
    os.makedirs("./outputs/results", exist_ok=True)
    df_results.to_csv("./outputs/results/benchmark_results.csv", index=False)
    print(f"\nResults saved to: ./outputs/results/benchmark_results.csv")

if __name__ == "__main__":
    run_benchmarks()
