"""Prefill + decode loop with KV cache for a patched HF Llama model."""

import torch
from torch import Tensor


class KVCache:
    """Pre-allocated key/value cache for all transformer layers.

    Pre-allocation avoids hot-path memory pressure. The append() method writes
    into the reserved space and returns valid slices for attention.
    """

    def __init__(
        self,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        max_seq: int,
        device: torch.device,
    ):
        self.pos = 0
        shape = (1, n_kv_heads, max_seq, head_dim)
        self.k = [torch.zeros(shape, dtype=torch.float16, device=device) for _ in range(n_layers)]
        self.v = [torch.zeros(shape, dtype=torch.float16, device=device) for _ in range(n_layers)]

    def append(
        self, layer: int, k: Tensor, v: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Write new K/V at current position; return valid slices for attention.

        Args:
            layer: layer index
            k, v:  [1, n_kv_heads, new_len, head_dim]

        Returns:
            (k_valid, v_valid): [1, n_kv_heads, pos+new_len, head_dim]
        """
        new_len = k.shape[2]
        self.k[layer][:, :, self.pos : self.pos + new_len, :] = k
        self.v[layer][:, :, self.pos : self.pos + new_len, :] = v
        end = self.pos + new_len
        return self.k[layer][:, :, :end, :], self.v[layer][:, :, :end, :]

    def advance(self, n: int = 1) -> None:
        self.pos += n


def _sample(logits: Tensor, temperature: float, top_k: int | None) -> Tensor:
    """Return next token id. logits: [1, vocab]."""
    if temperature == 0.0:
        return logits.argmax(dim=-1)  # greedy

    logits = logits / temperature
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def generate(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_k: int | None = None,
) -> str:
    """Prefill + decode loop.

    Uses the HF model's built-in past_key_values mechanism so RoPE and GQA
    attention are handled correctly. QuantLinear is a drop-in for nn.Linear,
    so no modifications to the attention code are needed.

    Args:
        model:          Llama model (fp16 weights + QuantLinear for transformer linears)
        tokenizer:      HF tokenizer
        prompt:         input string
        max_new_tokens: decode budget
        temperature:    0 → greedy; >0 → sampling
        top_k:          if set, restrict sampling to top-k logits

    Returns:
        generated text (not including the prompt)
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)

    eos_id = tokenizer.eos_token_id
    eos_ids: set[int] = (
        {eos_id} if isinstance(eos_id, int)
        else (set(eos_id) if eos_id is not None else set())
    )

    # Prefill: run full prompt, populate past_key_values
    with torch.no_grad():
        out = model(input_ids, use_cache=True)

    past_kv = out.past_key_values
    logits = out.logits[:, -1, :]  # [1, vocab]

    generated_ids: list[int] = []

    for _ in range(max_new_tokens):
        next_id = _sample(logits, temperature, top_k)
        token = next_id.item()

        if token in eos_ids:
            break

        generated_ids.append(token)

        # Decode step: one token, attend over cached K/V
        with torch.no_grad():
            out = model(
                next_id.unsqueeze(0),
                past_key_values=past_kv,
                use_cache=True,
            )

        past_kv = out.past_key_values
        logits = out.logits[:, -1, :]

    return tokenizer.decode(generated_ids, skip_special_tokens=True)
