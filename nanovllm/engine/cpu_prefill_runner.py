import torch

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.model_runner import get_model_cls
from nanovllm.utils.context import get_context, set_context, reset_context
from nanovllm.utils.loader import load_model


class CPUPrefillRunner:

    def __init__(self, config: Config):
        self.config = config
        self.block_size = config.kvcache_block_size
        hf_config = config.hf_config
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cpu")
        model_cls = get_model_cls(hf_config.model_type)
        self.model = model_cls(hf_config)
        load_model(self.model, config.model)
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

    @torch.inference_mode()
    def prefill(self, seq: Sequence) -> list[tuple[torch.Tensor, torch.Tensor]]:
        prompt_len = seq.num_prompt_tokens
        input_ids = torch.tensor(seq.token_ids, dtype=torch.int64)
        positions = torch.arange(prompt_len, dtype=torch.int64)
        cu = torch.tensor([0, prompt_len], dtype=torch.int32)
        set_context(
            True,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=prompt_len,
            max_seqlen_k=prompt_len,
            cpu_prefill_capture=True,
        )
        self.model(input_ids, positions)
        kv_layers = []
        for k, v in get_context().kv_captures:
            k = k.contiguous().cpu()
            v = v.contiguous().cpu()
            k = k.pin_memory()
            v = v.pin_memory()
            kv_layers.append((k, v))
        reset_context()
        return kv_layers
