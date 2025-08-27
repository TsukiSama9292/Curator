from typing import Any, Optional, List

try:
	from vllm import LLM, SamplingParams
	VLLM_AVAILABLE = True
except ImportError:
	VLLM_AVAILABLE = False
	class LLM:
		pass
	class SamplingParams:
		pass

from transformers import AutoTokenizer
from ray_curator.models.base import ModelInterface

class VLLMText(ModelInterface):
	"""Generic vLLM text generation model wrapper."""

	def __init__(
		self,
		model_id: str = None,
		model_dir: Optional[str] = None,
		tokenizer: Optional[str] = None,
		tokenizer_mode: str = "auto",
		skip_tokenizer_init: bool = False,
		trust_remote_code: bool = False,
		allowed_local_media_path: Optional[str] = None,
		tensor_parallel_size: int = 1,
		dtype: str = "auto",
		quantization: Optional[str] = None,
		revision: Optional[str] = None,
		tokenizer_revision: Optional[str] = None,
		seed: Optional[int] = None,
		gpu_memory_utilization: float = 0.90,
		swap_space: int = 4,
		cpu_offload_gb: int = 0,
		enforce_eager: bool = False,
		max_seq_len_to_capture: Optional[int] = None,
		disable_custom_all_reduce: bool = False,
		disable_async_output_proc: bool = False,
		hf_token: Optional[str] = None,
		hf_overrides: Optional[Any] = None,
		mm_processor_kwargs: Optional[dict] = None,
		override_pooler_config: Optional[Any] = None,
		compilation_config: Optional[Any] = None,
		# Sampling params
		max_output_tokens: int = 128,
		temperature: float = 0.7,
		top_p: float = 0.9,
		repetition_penalty: float = 1.0,
		stop_token_ids: Optional[List[int]] = None,
		# Pre
		pre_tokenizer: AutoTokenizer = None,
	):
		self.model_id = model_id
		self.model_dir = model_dir
		self.tokenizer_name = tokenizer
		self.tokenizer_mode = tokenizer_mode
		self.skip_tokenizer_init = skip_tokenizer_init
		self.trust_remote_code = trust_remote_code
		self.allowed_local_media_path = allowed_local_media_path if allowed_local_media_path is not None else ""
		self.tensor_parallel_size = tensor_parallel_size
		self.dtype = dtype
		self.quantization = quantization
		self.revision = revision
		self.tokenizer_revision = tokenizer_revision
		self.seed = seed
		self.gpu_memory_utilization = gpu_memory_utilization
		self.swap_space = swap_space
		self.cpu_offload_gb = cpu_offload_gb
		self.enforce_eager = enforce_eager
		self.max_seq_len_to_capture = max_seq_len_to_capture if max_seq_len_to_capture is not None else 2048
		self.disable_custom_all_reduce = disable_custom_all_reduce
		self.disable_async_output_proc = disable_async_output_proc
		self.hf_token = hf_token
		self.hf_overrides = hf_overrides
		self.mm_processor_kwargs = mm_processor_kwargs
		self.override_pooler_config = override_pooler_config
		self.compilation_config = compilation_config
		# Sampling params
		self.max_output_tokens = max_output_tokens
		self.temperature = temperature
		self.top_p = top_p
		self.repetition_penalty = repetition_penalty
		self.stop_token_ids = stop_token_ids
		self.tokenizer = pre_tokenizer if pre_tokenizer is not None else tokenizer

	def model_id_names(self) -> list[str]:
		return [self.model_id]

	def setup(self) -> None:
		if not VLLM_AVAILABLE:
			raise ImportError("vllm is required for VLLMText model but is not installed. Please install vllm: pip install vllm")

		# Model path: if model_dir is set, use it, else use model_id
		if self.model_dir:
			model_path = self.model_dir
		else:
			model_path = self.model_id

		self.llm = LLM(
			model=model_path,
			tokenizer=self.tokenizer_name,
			tokenizer_mode=self.tokenizer_mode,
			skip_tokenizer_init=self.skip_tokenizer_init,
			trust_remote_code=self.trust_remote_code,
			allowed_local_media_path=self.allowed_local_media_path,
			tensor_parallel_size=self.tensor_parallel_size,
			dtype=self.dtype,
			quantization=self.quantization,
			revision=self.revision,
			tokenizer_revision=self.tokenizer_revision,
			seed=self.seed,
			gpu_memory_utilization=self.gpu_memory_utilization,
			swap_space=self.swap_space,
			cpu_offload_gb=self.cpu_offload_gb,
			enforce_eager=self.enforce_eager,
			max_seq_len_to_capture=self.max_seq_len_to_capture,
			disable_custom_all_reduce=self.disable_custom_all_reduce,
			disable_async_output_proc=self.disable_async_output_proc,
			hf_token=self.hf_token,
			hf_overrides=self.hf_overrides,
			mm_processor_kwargs=self.mm_processor_kwargs,
			override_pooler_config=self.override_pooler_config,
			compilation_config=self.compilation_config,
		)
		self.sampling_params = SamplingParams(
			temperature=self.temperature,
			top_p=self.top_p,
			repetition_penalty=self.repetition_penalty,
			max_tokens=self.max_output_tokens,
			stop_token_ids=self.stop_token_ids,
		)
		self.tokenizer = AutoTokenizer.from_pretrained(model_path) if self.tokenizer is None else self.tokenizer

	def generate(self, inputs: list[Any], use_chat_template: bool = True) -> list[str]:
		"""
		Args:
			inputs: list of str (for plain text) or list of dict (for chat)
			use_chat_template: if True, use tokenizer.apply_chat_template
		Returns:
			list of generated strings
		"""
		if use_chat_template:
			formatted_inputs = self.tokenizer.apply_chat_template(inputs, tokenize=False, add_generation_prompt=True)
		else:
			formatted_inputs = inputs
		results = self.llm.generate(formatted_inputs, sampling_params=self.sampling_params)
		return [result.outputs[0].text for result in results]

if __name__ == "__main__":
	# Example usage for VLLMText
    # if GPU VRAM not enough
    # nano ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/eb25fbe4f35f7147763bc24445679d1c00588d89/config.json
    # change the "max_position_embeddings": 2048 (or less)
	model = VLLMText(
		model_id="Qwen/Qwen3-4B-Instruct-2507",  # You can replace with any HuggingFace model
		max_output_tokens=64,
		temperature=0.7,
		top_p=0.9,
		quantization=None,
		trust_remote_code=True,
		gpu_memory_utilization=0.4,
	)
	model.setup()
	# Plain text generation
	prompts = ["Translate to Traditional Chinese: Hello World"]
	outputs = model.generate(prompts, use_chat_template=False)
	print("Text outputs:", outputs)

	# Chat mode (if chat template is supported)
	chat_inputs = [
		[
			{"role": "system", "content": "You are a helpful assistant."},
			{"role": "user", "content": "Translate to Traditional Chinese: Hello World"}
		]
	]
	chat_outputs = model.generate(chat_inputs)
	print("Chat outputs:", chat_outputs)