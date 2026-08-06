"""Unit tests for core modules."""

import json
import torch
import pytest
import tempfile
from pathlib import Path

from isla.model import IslaModel, SpikingBlock
from isla.model.neurons import LIFNeuron, spike_fn
from isla.model.attention import SpikeSyncAttention, RotaryEmbedding
from isla.config import ModelConfig, TrainConfig


def _cfg(**kw):
    defaults = dict(vocab_size=256, hidden_dim=64, num_layers=2, num_heads=2,
                    num_timesteps=2, max_seq_len=32, ff_mult=4, dropout=0.0)
    defaults.update(kw)
    return ModelConfig(**defaults)


class TestNeurons:
    def test_spikes_are_binary(self):
        out = spike_fn(torch.randn(10), threshold=0.0)
        assert set(out.unique().tolist()).issubset({0.0, 1.0})

    def test_gradient_flows(self):
        v = torch.randn(10, requires_grad=True)
        spike_fn(v, threshold=0.0).sum().backward()
        assert v.grad is not None and (v.grad != 0).any()

    def test_lif_shapes(self):
        lif = LIFNeuron(32)
        s, m, _ = lif.step(torch.randn(4, 32))
        assert s.shape == (4, 32) and m.shape == (4, 32)

    def test_beta_constrained(self):
        assert (LIFNeuron(8, beta=0.95).beta > 0).all()

    def test_multi_step(self):
        """multi_step returns per-unit rates with correct shape."""
        lif = LIFNeuron(64)
        current = torch.randn(4, 16, 64)
        spike_sum, membrane, rate_per_unit = lif.multi_step(current, T=4)
        assert spike_sum.shape == (4, 16, 64)
        assert membrane.shape == (4, 16, 64)
        assert rate_per_unit.shape == (64,)
        assert (rate_per_unit >= 0).all() and (rate_per_unit <= 1).all()


class TestAttention:
    def test_output_shape(self):
        attn = SpikeSyncAttention(64, 2)
        out, _ = attn(torch.randn(2, 8, 64))
        assert out.shape == (2, 8, 64)

    def test_causal_mask(self):
        attn = SpikeSyncAttention(64, 2)
        mask = torch.triu(torch.full((8, 8), float("-inf")), diagonal=1)
        out, _ = attn(torch.randn(2, 8, 64), mask=mask)
        assert out.shape == (2, 8, 64)

    def test_tau_positive(self):
        assert SpikeSyncAttention(32, 2).tau.item() > 0

    def test_rope_shapes(self):
        rope = RotaryEmbedding(32, max_seq_len=64)
        cos, sin = rope(seq_len=16, offset=0)
        assert cos.shape == (1, 1, 16, 32)
        assert sin.shape == (1, 1, 16, 32)

    def test_rope_offset(self):
        """RoPE with offset should produce same values as full sequence slice."""
        rope = RotaryEmbedding(32, max_seq_len=64)
        cos_full, sin_full = rope(seq_len=32, offset=0)
        cos_off, sin_off = rope(seq_len=16, offset=16)
        assert torch.allclose(cos_full[:, :, 16:, :], cos_off, atol=1e-6)
        assert torch.allclose(sin_full[:, :, 16:, :], sin_off, atol=1e-6)

    def test_standard_attention_cache_matches_full_forward(self):
        from isla.inference.generate import _create_caches

        model = IslaModel(_cfg(use_standard_attention=True)).eval()
        tokens = torch.randint(0, 256, (1, 8))
        with torch.no_grad():
            full, _, _ = model(tokens)
            caches = _create_caches(model)
            prefill, _, caches = model(tokens[:, :5], caches=caches)
            steps = []
            for position in range(5, tokens.shape[1]):
                logits, _, caches = model(tokens[:, position:position + 1], caches=caches)
                steps.append(logits)

        cached = torch.cat([prefill, *steps], dim=1)
        assert torch.allclose(full, cached, atol=1e-4)


class TestModel:
    def test_forward(self):
        cfg = _cfg()
        m = IslaModel(cfg)
        logits, metrics, _ = m(torch.randint(0, 256, (2, 16)))
        assert logits.shape == (2, 16, 256)
        assert 0.0 <= metrics["mean_spike_rate"].item() <= 1.0
        assert "spike_rate_std" in metrics

    def test_backward(self):
        m = IslaModel(_cfg())
        logits, _, _ = m(torch.randint(0, 256, (2, 16)))
        logits.sum().backward()
        for n, p in m.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"no gradient: {n}"

    def test_weight_tying(self):
        m = IslaModel(_cfg())
        assert m.lm_head.weight is m.token_emb.weight

    def test_no_pos_emb(self):
        """Model uses RoPE, not absolute position embeddings."""
        m = IslaModel(_cfg())
        assert not hasattr(m, "pos_emb"), "pos_emb should be removed (using RoPE)"

    def test_identity_mlp_residual_preserves_hidden_state(self):
        class ZeroAttention(torch.nn.Module):
            def forward(self, x, mask=None, cache=None):
                return torch.zeros_like(x), cache

        class UnitMLP(torch.nn.Module):
            def forward(self, x):
                return torch.ones_like(x), torch.zeros(x.shape[-1])

        block = SpikingBlock(_cfg(mlp_residual_mode="identity"))
        block.attn = ZeroAttention()
        block.mlp = UnitMLP()
        x = torch.randn(2, 4, 64)

        out, _, _ = block(x)

        assert torch.allclose(out, x + block.gate * torch.ones_like(x))

    def test_save_load_roundtrip(self):
        m = IslaModel(_cfg())
        ids = torch.randint(0, 256, (1, 8))
        with torch.no_grad():
            out1, _, _ = m(ids)
        with tempfile.TemporaryDirectory() as d:
            m.save_pretrained(d)
            m2 = IslaModel.from_pretrained(d)
            with torch.no_grad():
                out2, _, _ = m2(ids)
        assert torch.allclose(out1, out2, atol=1e-5)

    def test_overfit(self):
        """Loss must decrease when training on 8 fixed sequences for 50 steps."""
        m = IslaModel(_cfg(spike_reg_lambda=0.0))
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        crit = torch.nn.CrossEntropyLoss()
        data = torch.randint(0, 256, (8, 16))
        losses = []
        for _ in range(50):
            logits, _, _ = m(data)
            loss = crit(logits[:, :-1].contiguous().view(-1, 256), data[:, 1:].contiguous().view(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5 * 0.8


class TestConfig:
    def test_bf16_fp16_mutual_exclusion(self):
        """Setting both bf16 and fp16 should raise ValueError."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            TrainConfig(bf16=True, fp16=True)

    def test_bf16_only_ok(self):
        TrainConfig(bf16=True, fp16=False)

    def test_fp16_only_ok(self):
        TrainConfig(bf16=False, fp16=True)

    def test_attention_variants_are_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            _cfg(use_standard_attention=True, use_spike_ssm=True)

    def test_invalid_residual_mode_is_rejected(self):
        with pytest.raises(ValueError, match="mlp_residual_mode"):
            _cfg(mlp_residual_mode="unknown")

    def test_legacy_config_keeps_spike_first_residual(self):
        legacy = {
            "vocab_size": 256,
            "hidden_dim": 64,
            "num_layers": 2,
            "num_heads": 2,
            "num_timesteps": 2,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model_config.json"
            path.write_text(json.dumps(legacy))
            loaded = ModelConfig.load(path)

        assert loaded.mlp_residual_mode == "spike_first"



class TestSlopeScheduling:
    def test_slope_scheduling(self):
        """Verify that dynamic surrogate slope scheduling correctly computes and injects slopes."""
        from isla.training.trainer import IslaTrainer
        
        m_cfg = ModelConfig(vocab_size=256, hidden_dim=64, num_layers=2, num_heads=2, num_timesteps=2)
        t_cfg = TrainConfig(
            surrogate_slope=5.0,
            surrogate_slope_final=15.0,
            max_steps=10,
            warmup_steps=0,
            bf16=False,
            fp16=False
        )
        
        m = IslaModel(m_cfg)
        trainer = IslaTrainer(m, t_cfg, m_cfg)
        
        # Step 0: Should be initial slope (5.0)
        trainer.step = 0
        slope_0 = trainer._update_surrogate_slope()
        assert abs(slope_0 - 5.0) < 1e-4
        for module in m.modules():
            if hasattr(module, "slope"):
                assert abs(module.slope - 5.0) < 1e-4
                
        # Step 5: Halfway (progress = 0.5), Cosine schedule should be exactly 10.0
        trainer.step = 5
        slope_5 = trainer._update_surrogate_slope()
        assert abs(slope_5 - 10.0) < 1e-4
        for module in m.modules():
            if hasattr(module, "slope"):
                assert abs(module.slope - 10.0) < 1e-4
                
        # Step 10: Complete (progress = 1.0), should be final slope (15.0)
        trainer.step = 10
        slope_10 = trainer._update_surrogate_slope()
        assert abs(slope_10 - 15.0) < 1e-4
        for module in m.modules():
            if hasattr(module, "slope"):
                assert abs(module.slope - 15.0) < 1e-4


class TestSSM:
    def test_ssm_does_not_allocate_dense_causal_mask(self, monkeypatch):
        cfg = _cfg(use_spike_ssm=True)
        model = IslaModel(cfg).eval()

        def fail_if_called(*args, **kwargs):
            raise AssertionError("SSM forward must not construct an O(L^2) causal mask")

        monkeypatch.setattr(torch, "triu", fail_if_called)
        with torch.no_grad():
            logits, _, _ = model(torch.randint(0, 256, (1, 8)))

        assert logits.shape == (1, 8, 256)

    def test_ssm_equivalence(self):
        """SSM with cache (incremental decoding) must match standard forward pass without cache."""
        from isla.inference.generate import _create_caches
        cfg = ModelConfig(vocab_size=256, hidden_dim=64, num_layers=2, num_heads=2,
                          num_timesteps=2, max_seq_len=32, use_spike_ssm=True)
        m = IslaModel(cfg)
        m.eval()
        
        # Sequence of length 5 (prefill) + 3 tokens (decoding steps)
        x_prefill = torch.randint(0, 256, (1, 5))
        x_steps = torch.randint(0, 256, (1, 3))
        x_full = torch.cat([x_prefill, x_steps], dim=1)
        
        # 1. Full un-cached forward pass
        with torch.no_grad():
            logits_full, _, _ = m(x_full)
            
        # 2. Cached forward pass (Prefill + step-by-step decoding)
        caches = _create_caches(m)
        
        # Prefill
        with torch.no_grad():
            logits_cached_prefill, _, caches = m(x_prefill, caches=caches)
            
        # Verify prefill logits match the full logits at corresponding steps
        assert torch.allclose(logits_full[:, :5], logits_cached_prefill, atol=1e-4)
        
        # Step-by-step decoding
        decoded_logits = []
        for i in range(3):
            token = x_steps[:, i:i+1]
            with torch.no_grad():
                step_logits, _, caches = m(token, caches=caches)
                decoded_logits.append(step_logits)
                
        logits_cached_decoded = torch.cat(decoded_logits, dim=1)
        
        # Verify step-by-step decoding logits match the full logits at corresponding steps
        assert torch.allclose(logits_full[:, 5:], logits_cached_decoded, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
