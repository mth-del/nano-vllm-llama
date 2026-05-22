import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.cpu_prefill_runner import CPUPrefillRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.cpu_prefill_runner = CPUPrefillRunner(config) if config.pd_separation else None
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self.config = config
        self.compress_step_count = 0
        atexit.register(self.exit)

    def exit(self):
        if not hasattr(self, "model_runner"):
            return
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        if self.config.pd_separation:
            return self._step_pd()
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids, compression_events = self.model_runner.call("run", seqs, is_prefill)
        if compression_events:
            self.compress_step_count += len(compression_events)
        self.scheduler.postprocess(seqs, token_ids, is_prefill, compression_events)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def _step_pd(self):
        outputs = []
        num_tokens = 0

        # 1) CPU prefill：waiting → prefill_ready（不占 GPU block）
        cpu_seqs = self.scheduler.schedule_cpu_prefill()
        if cpu_seqs:
            for seq in cpu_seqs:
                seq.cpu_kv_layers = self.cpu_prefill_runner.prefill(seq)
                self.scheduler.prefill_ready.append(seq)
            num_tokens += sum(seq.num_prompt_tokens for seq in cpu_seqs)

        # 2) GPU handoff：prefill_ready → allocate + import_kv → running
        handoff_seqs = self.scheduler.schedule_gpu_handoff()
        if handoff_seqs:
            for seq in handoff_seqs:
                cpu_kv = seq.cpu_kv_layers
                assert cpu_kv is not None
                self.model_runner.call("import_kv", seq, cpu_kv)
                seq.cpu_kv_layers = None
                seq.num_cached_tokens = seq.num_prompt_tokens
            handoff_tokens = sum(seq.num_prompt_tokens for seq in handoff_seqs)
            num_tokens += handoff_tokens
            token_ids, compression_events = self.model_runner.call("run", handoff_seqs, False)
            if compression_events:
                self.compress_step_count += len(compression_events)
            self.scheduler.postprocess(handoff_seqs, token_ids, False, compression_events)
            outputs.extend(
                (seq.seq_id, seq.completion_token_ids) for seq in handoff_seqs if seq.is_finished
            )

        if cpu_seqs or handoff_seqs:
            return outputs, num_tokens if num_tokens > 0 else sum(s.num_prompt_tokens for s in cpu_seqs)

        # 3) Decode only（GPU 上仅 running 序列持有 KV）
        seqs, is_prefill = self.scheduler.schedule()
        assert not is_prefill
        num_tokens = -len(seqs)
        token_ids, compression_events = self.model_runner.call("run", seqs, False)
        if compression_events:
            self.compress_step_count += len(compression_events)
        self.scheduler.postprocess(seqs, token_ids, False, compression_events)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
                "PD_ready": len(self.scheduler.prefill_ready),
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
