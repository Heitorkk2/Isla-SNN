"""Training loop with JSONL + optional W&B logging.

Loss:  L = CrossEntropy + λ · spike_penalty
    where spike_penalty = |rate - target|² (if target > 0) or rate (raw penalisation)
Schedule:  linear warmup → cosine decay
Precision:  bf16 or fp16 via torch AMP
Logging:  train_log.jsonl + wandb (if configured)
"""

import json
import time
import math
from pathlib import Path
from contextlib import nullcontext

from tqdm.auto import tqdm
import torch
import torch.nn as nn

from isla.config import ModelConfig, TrainConfig

try:
    import wandb as _wandb
except ImportError:
    _wandb = None


def _get_amp_context(device, amp_dtype, enabled):
    """Return the correct autocast context for any PyTorch version."""
    if not enabled or amp_dtype is None:
        return nullcontext()
    try:
        return torch.amp.autocast(device_type=device.type, dtype=amp_dtype)
    except (TypeError, AttributeError):
        return torch.cuda.amp.autocast(dtype=amp_dtype)


def _get_scaler(use_fp16):
    """Return GradScaler compatible with any PyTorch version."""
    try:
        return torch.amp.GradScaler("cuda", enabled=use_fp16)
    except (TypeError, AttributeError):
        return torch.cuda.amp.GradScaler(enabled=use_fp16)


def _cosine_lr(step, warmup, total, lr_max, lr_min):
    if step < warmup:
        return lr_max * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


class IslaTrainer:
    """Handles the full training lifecycle: forward, backward, logging, checkpointing."""

    def __init__(self, model, train_cfg: TrainConfig, model_cfg: ModelConfig):
        self.model = model
        self.cfg = train_cfg
        self.model_cfg = model_cfg
        self.device = next(model.parameters()).device

        # keep reference to the unwrapped model (torch.compile wraps it)
        self._raw_model = getattr(model, "_orig_mod", model)

        self.out = Path(self.cfg.checkpoint.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # separate weight-decay and no-decay groups
        wd_params, no_wd_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if any(k in name for k in ("bias", "norm", "emb", "beta_raw", "tau_raw", "gate_raw", "alpha_raw", "adaptation_strength")):
                no_wd_params.append(p)
            else:
                wd_params.append(p)

        self.optimizer = torch.optim.AdamW([
            {"params": wd_params, "weight_decay": self.cfg.weight_decay},
            {"params": no_wd_params, "weight_decay": 0.0},
        ], lr=self.cfg.lr, betas=(0.9, 0.95))

        # AMP
        self.amp_dtype = torch.bfloat16 if self.cfg.bf16 else (torch.float16 if self.cfg.fp16 else None)
        self.use_amp = self.amp_dtype is not None
        self.scaler = _get_scaler(self.cfg.fp16)

        self.step = 0
        self.tokens_seen = 0
        self.best_val_loss = float("inf")

        # gradient checkpointing
        if train_cfg.gradient_checkpointing:
            self._raw_model.enable_gradient_checkpointing()
            print("[TRAIN] gradient checkpointing enabled")

        # persist configs
        model_cfg.save(self.out / "model_config.json")
        self.cfg.save(self.out / "train_config.json")

        # JSONL log (append if resuming, new otherwise)
        self.log_path = self.out / "train_log.jsonl"
        resuming = bool(self.cfg.checkpoint.resume_from)
        if not resuming:
            self.log_path.write_text("")

        # resume from checkpoint
        if resuming:
            self._resume(self.cfg.checkpoint.resume_from)

        # W&B init (after resume so step is correct)
        self._wandb_run = None
        self._init_wandb()

    def _init_wandb(self):
        """Initialise W&B run if configured and the library is installed."""
        wcfg = self.cfg.wandb
        if not wcfg.enabled:
            return
        if _wandb is None:
            print("[WARN] wandb not installed, skipping. pip install wandb")
            return

        from dataclasses import asdict
        run_name = wcfg.run_name or f"isla-{self.model_cfg.hidden_dim}d-{self.model_cfg.num_layers}L"

        self._wandb_run = _wandb.init(
            project=wcfg.project,
            name=run_name,
            config={
                "model": asdict(self.model_cfg),
                "train": asdict(self.cfg),
            },
            resume="allow" if self.cfg.checkpoint.resume_from else None,
        )
        _wandb.define_metric("train/step")
        _wandb.define_metric("train/*", step_metric="train/step")
        _wandb.define_metric("eval/*", step_metric="train/step")
        print(f"[WANDB] run={self._wandb_run.name}  project={wcfg.project}")

    def _wandb_log(self, data, step):
        """Log to W&B if active. No-op otherwise."""
        if self._wandb_run is not None:
            _wandb.log(data, step=step)

    def _log(self, entry):
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _update_lr(self):
        lr = _cosine_lr(self.step, self.cfg.warmup_steps, self.cfg.max_steps, self.cfg.lr, self.cfg.min_lr)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def _resume(self, ckpt_dir):
        """Restore model weights, optimizer state, scaler, and step counter."""
        d = Path(ckpt_dir)
        weights = d / "model.pth"
        state_file = d / "training_state.pth"

        if weights.exists():
            self._raw_model.load_state_dict(torch.load(weights, map_location=self.device, weights_only=True))

        if state_file.exists():
            state = torch.load(state_file, map_location=self.device, weights_only=False)
            self.step = state.get("step", 0)
            self.tokens_seen = state.get("tokens", 0)
            self.best_val_loss = state.get("best_val_loss", float("inf"))
            self.optimizer.load_state_dict(state["optimizer"])
            self.scaler.load_state_dict(state["scaler"])

        print(f"[TRAIN] resumed from {ckpt_dir} at step {self.step} ({self.tokens_seen:,} tokens)")

    def _forward_backward(self, batch):
        ids = batch["input_ids"].to(self.device)
        labels = batch.get("labels", ids).to(self.device)

        with _get_amp_context(self.device, self.amp_dtype, self.use_amp):
            logits, metrics, _ = self.model(ids)
            # Flattening to 2D forces the memory pointer to be fully contiguous.
            # Hopper (H100) and Blackwell (5070) bfloat16 ATen kernels crash with Illegal Memory Access
            # if passed a transposed 3D dimension due to unaligned CUDA warp boundaries.
            shift_logits = logits[:, :-1].contiguous().view(-1, self.model_cfg.vocab_size)
            shift_labels = labels[:, 1:].contiguous().view(-1)
            ce = self.criterion(shift_logits, shift_labels)

            # spike penalty: target rate or raw rate
            rate = metrics["mean_spike_rate"]
            if self.model_cfg.target_spike_rate > 0:
                diff = rate - self.model_cfg.target_spike_rate
                spike_pen = self.model_cfg.spike_reg_lambda * (diff * diff)
            else:
                spike_pen = self.model_cfg.spike_reg_lambda * rate

            loss = (ce + spike_pen) / self.cfg.gradient_accumulation_steps

        self.scaler.scale(loss).backward()

        # count real tokens (excluding padding / -100 labels)
        real_tokens = (labels != -100).sum().item()

        return {"loss": ce.item(), "spike_rate": rate.item(),
                "spike_rate_std": metrics["spike_rate_std"].item(),
                "real_tokens": real_tokens}

    def _collect_diagnostics(self):
        """Gather τ, β, and neuron health from the model for logging."""
        diag = {}
        total_dead, total_saturated, total_neurons = 0, 0, 0

        for i, block in enumerate(self._raw_model.blocks):
            attn = block.attn
            if hasattr(attn, "tau"):
                diag[f"diag/tau_layer_{i}"] = attn.tau.item()
            lif = block.mlp.lif
            beta = lif.beta.mean().item()
            diag[f"diag/beta_mean_layer_{i}"] = beta

        # neuron health from last forward pass spike rates
        for i, block in enumerate(self._raw_model.blocks):
            # spike_rates_per_layer has shape (d_ff,) — per-unit mean rate
            # we check which units are dead (<1%) or saturated (>95%)
            rates = getattr(block.mlp, '_last_rates', None)
            if rates is not None:
                n = rates.numel()
                dead = (rates < 0.01).sum().item()
                saturated = (rates > 0.95).sum().item()
                total_dead += dead
                total_saturated += saturated
                total_neurons += n

        if total_neurons > 0:
            diag["diag/dead_neurons_pct"] = total_dead / total_neurons * 100
            diag["diag/saturated_neurons_pct"] = total_saturated / total_neurons * 100

        # v3 diagnostics: gate values, alpha (membrane mixing), SFA adaptation
        gate_values = []
        alpha_values = []
        adapt_values = []
        for i, block in enumerate(self._raw_model.blocks):
            if hasattr(block, "gate"):
                g = block.gate.item()
                diag[f"diag/gate_layer_{i}"] = g
                gate_values.append(g)
            if hasattr(block.mlp, "alpha"):
                a = block.mlp.alpha.item()
                diag[f"diag/alpha_layer_{i}"] = a
                alpha_values.append(a)
            if hasattr(block.mlp.lif, "adaptation_strength"):
                ad = torch.relu(block.mlp.lif.adaptation_strength).mean().item()
                diag[f"diag/sfa_strength_layer_{i}"] = ad
                adapt_values.append(ad)
        if gate_values:
            diag["diag/gate_mean"] = sum(gate_values) / len(gate_values)
        if alpha_values:
            diag["diag/alpha_mean"] = sum(alpha_values) / len(alpha_values)
        if adapt_values:
            diag["diag/sfa_strength_mean"] = sum(adapt_values) / len(adapt_values)

        return diag

    @torch.no_grad()
    def _evaluate(self, val_loader, max_batches=50):
        self.model.eval()
        total_loss, total_spike, count = 0.0, 0.0, 0
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            with _get_amp_context(self.device, self.amp_dtype, self.use_amp):
                logits, metrics, _ = self.model(ids)
                total_loss += self.criterion(
                    logits[:, :-1].transpose(1, 2),
                    labels[:, 1:],
                ).item()
                total_spike += metrics["mean_spike_rate"].item()
            count += 1
        self.model.train()
        return {
            "val_loss": total_loss / max(count, 1),
            "val_spike_rate": total_spike / max(count, 1),
        }

    def _save(self, tag):
        d = self.out / tag
        d.mkdir(parents=True, exist_ok=True)
        self.model_cfg.save(d / "model_config.json")
        torch.save(self._raw_model.state_dict(), d / "model.pth")
        torch.save({"step": self.step, "tokens": self.tokens_seen,
                     "best_val_loss": self.best_val_loss,
                     "optimizer": self.optimizer.state_dict(),
                     "scaler": self.scaler.state_dict()}, d / "training_state.pth")

    def train(self, train_loader, val_loader=None):
        self.model.train()
        self.optimizer.zero_grad()
        t0 = time.time()

        eff = self.cfg.effective_batch_size

        # auto-compute max_steps from epochs if not set
        steps_per_epoch = len(train_loader) // self.cfg.gradient_accumulation_steps
        if self.cfg.max_steps <= 0:
            self.cfg.max_steps = steps_per_epoch * self.cfg.num_epochs

        print(f"[TRAIN] epochs={self.cfg.num_epochs}  steps={self.cfg.max_steps:,}  "
              f"steps/epoch={steps_per_epoch:,}  eff_batch={eff}")
        print(f"[TRAIN] params={self._raw_model.count_params():,}  "
              f"amp={'bf16' if self.cfg.bf16 else 'fp16' if self.cfg.fp16 else 'off'}  log={self.log_path}")

        epoch = 0
        it = iter(train_loader)

        pbar = tqdm(total=self.cfg.max_steps, initial=self.step, 
                    desc=f"Epoch {epoch+1}/{self.cfg.num_epochs} [Train]", dynamic_ncols=True)

        try:
            while self.step < self.cfg.max_steps:
                lr = self._update_lr()
                acc = {"loss": 0.0, "spike_rate": 0.0, "spike_rate_std": 0.0, "real_tokens": 0}

                for _ in range(self.cfg.gradient_accumulation_steps):
                    try:
                        batch = next(it)
                    except StopIteration:
                        epoch += 1
                        pbar.set_description(f"Epoch {epoch+1}/{self.cfg.num_epochs} [Train]")
                        it = iter(train_loader)
                        batch = next(it)
                    m = self._forward_backward(batch)
                    acc["loss"] += m["loss"] / self.cfg.gradient_accumulation_steps
                    acc["spike_rate"] += m["spike_rate"] / self.cfg.gradient_accumulation_steps
                    acc["spike_rate_std"] += m["spike_rate_std"] / self.cfg.gradient_accumulation_steps
                    acc["real_tokens"] += m["real_tokens"]

                # clip gradients and capture grad norm for logging
                self.scaler.unscale_(self.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm).item()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                self.step += 1
                self.tokens_seen += acc["real_tokens"]

                # train logging
                if self.step % self.cfg.log_every == 0:
                    dt = time.time() - t0
                    tps = acc["real_tokens"] * (self.cfg.log_every / max(dt, 1e-6))
                    ppl = math.exp(min(acc["loss"], 20))

                    entry = {
                        "step": self.step, "tokens_seen": self.tokens_seen,
                        "loss": round(acc["loss"], 4),
                        "perplexity": round(ppl, 2),
                        "spike_rate": round(acc["spike_rate"], 4),
                        "spike_rate_std": round(acc["spike_rate_std"], 4),
                        "grad_norm": round(grad_norm, 4),
                        "lr": round(lr, 8), "tokens_per_sec": round(tps, 1),
                    }
                    self._log(entry)
                    self._wandb_log({
                        "train/step": self.step,
                        "train/loss": acc["loss"],
                        "train/perplexity": ppl,
                        "train/spike_rate": acc["spike_rate"],
                        "train/spike_rate_std": acc["spike_rate_std"],
                        "train/grad_norm": grad_norm,
                        "train/lr": lr,
                        "train/tokens_per_sec": tps,
                        "train/tokens_seen": self.tokens_seen,
                        **self._collect_diagnostics(),
                    }, step=self.step)

                    pbar.set_postfix({
                        "loss": f"{entry['loss']:.4f}",
                        "ppl": f"{ppl:.0f}",
                        "spike": f"{entry['spike_rate']:.3f}",
                        "tok/s": f"{tps:,.0f}"
                    })
                    t0 = time.time()

                # eval logging + best model tracking
                if val_loader and self.step % self.cfg.eval_every == 0:
                    val = self._evaluate(val_loader)
                    val_ppl = math.exp(min(val["val_loss"], 20))

                    self._log({"step": self.step, "val_loss": round(val["val_loss"], 4),
                               "val_ppl": round(val_ppl, 2), "val_spike_rate": round(val["val_spike_rate"], 4)})
                    self._wandb_log({
                        "train/step": self.step,
                        "eval/val_loss": val["val_loss"],
                        "eval/val_perplexity": val_ppl,
                        "eval/val_spike_rate": val["val_spike_rate"],
                    }, step=self.step)

                    improved = val["val_loss"] < self.best_val_loss
                    if improved:
                        self.best_val_loss = val["val_loss"]
                        if self.cfg.checkpoint.save_best:
                            self._save("best")

                    marker = " ★ best" if improved else ""
                    # tqdm.write ensures it prints cleanly above the progress bar
                    tqdm.write(f"  [EVAL] step {self.step:,} | val_loss {val['val_loss']:.4f} | val_ppl {val_ppl:,.0f}{marker}")
                    t0 = time.time()

                # periodic checkpoint
                if self.step % self.cfg.checkpoint.save_every == 0:
                    self._save(f"step_{self.step}")
                    self._save("latest")
                    tqdm.write(f"  [CKPT] saved step_{self.step}/ and latest/")

                pbar.update(1)

        except KeyboardInterrupt:
            tqdm.write(f"\n[TRAIN] interrupted at step {self.step:,}")

        finally:
            pbar.close()
            # always save latest state (even on crash/interrupt)
            self._save("latest")
            print(f"\n[CKPT] saved latest/ (step {self.step:,})")

            if self.cfg.checkpoint.save_final:
                self._save("final")
                print(f"[CKPT] saved final/")

            if self._wandb_run is not None:
                _wandb.finish()

            print(f"[TRAIN] Done. {self.step:,} steps, {self.tokens_seen:,} tokens processed.")
