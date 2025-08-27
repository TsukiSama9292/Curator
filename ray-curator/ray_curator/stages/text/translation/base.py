import yaml
import re
from ray_curator.stages.text.translation.prompt import (
    INITIAL_TRANSLATION_PROMPT,
    REFLECTION_COUNTRY_TRANSLATION_PROMPT,
    REFLECTION_TRANSLATION_PROMPT,
    IMPROVE_TRANSLATION_PROMPT,
)
from openai import OpenAI
from transformers import AutoTokenizer
import pandas as pd

def extract_content(s:str, debug:bool=False) -> str|None:
    match = re.search(r'<IMPROVED_TRANSLATION>(.*?)</IMPROVED_TRANSLATION>', s, re.DOTALL)
    if match:
        extracted = match.group(1).strip()
        if debug:
            print(f"EXTRACTED: {extracted}")
        return extracted
    else:
        if "</IMPROVED_TRANSLATION>" in s:
            s = s.replace("</IMPROVED_TRANSLATION>", "")
            if debug:
                print(f"EXTRACTED (no closing tag): remove </IMPROVED_TRANSLATION>")
        elif "<IMPROVED_TRANSLATION>" in s:
            s = s.replace("<IMPROVED_TRANSLATION>", "")
            if debug:
                print(f"EXTRACTED (no opening tag): remove <IMPROVED_TRANSLATION>")
        else:
            if debug:
                print("No tags found.")
        return s

class TextSplitter:
    """
    Utility class for splitting text into chunks based on token count.
    """
    def __init__(self, model_name: str = "", hf_token: str = None, max_token_per_chunk: int = 4096, tokenizer: AutoTokenizer = None, spacy_model: str = "en_core_web_lg"):
        import spacy
        self.model_name = model_name
        self.hf_token = hf_token
        self.max_token_per_chunk = max_token_per_chunk
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            import subprocess
            print(f"spaCy model '{spacy_model}' not found. Downloading...")
            subprocess.run(["python", "-m", "spacy", "download", spacy_model], check=True)
            self.nlp = spacy.load(spacy_model)
        # Load tokenizer for counting tokens
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif self.model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.hf_token, use_fast=True)
        else:
            raise ValueError("Model name is empty. Please provide a valid model name.")

    def split_sentences(self, text: str) -> list[str]:
        # split sentences by spaCy
        doc = self.nlp(text)
        return [sent.text for sent in doc.sents]

    def num_tokens_in_string_hf(self, input_str: str) -> int:
        # Count tokens in a string using the tokenizer
        return len(self.tokenizer.encode(input_str))

    def split_long_text(self, text: str, max_tokens: int = None) -> list[str]:
        # Split long text into chunks based on max token count
        if max_tokens is None:
            max_tokens = self.max_token_per_chunk
        
        all_token_count = self.num_tokens_in_string_hf(text)
        if all_token_count <= max_tokens:
            return [text]
        
        sentences = self.split_sentences(text)
        chunks = []
        current_chunk = ''
        for sentence in sentences:
            # Separate sentences with spaces
            test_chunk = current_chunk + " " + sentence
            token_count = self.num_tokens_in_string_hf(test_chunk)
            if token_count > max_tokens:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
            else:
                current_chunk = test_chunk
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

class TranslationDataGenerator():
    """
    Synthetic data generator for translation tasks.
    Supports both function-based and YAML-based configuration.
    """
    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        hf_model: str = None,
        hf_token: str = None,
        api_model: str = None,
        max_token_per_chunk: int = 5000,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 24576,
        stop: list[str] = None,
        source_lang: str = "English",
        target_lang: str = "Traditional Chinese",
        country: str = "Taiwan",
        spacy_model: str = "en_core_web_lg",
        gpu_memory_utilization: float = 0.4,
        quantization: list[str] = None,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.api_model = api_model
        self.hf_model = hf_model
        self.hf_token = hf_token
        self.max_token_per_chunk = max_token_per_chunk
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.country = country
        self.spacy_model = spacy_model
        self.quantization = quantization
        self.gpu_memory_utilization = gpu_memory_utilization
        self.stop = stop
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model, use_auth_token=self.hf_token)
        if self.stop is None:
            self.stop = [self.tokenizer.decode([self.tokenizer.eos_token_id])]
        
        if self.base_url:
            self.openai_client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        elif self.hf_model:
            from ray_curator.models.vllm_text import VLLMText
            self.llm = VLLMText(
                model_id=self.hf_model,
                max_output_tokens=self.max_tokens,
                temperature=self.temperature,
                hf_token=self.hf_token,
                top_p=self.top_p,
                quantization=self.quantization,
                trust_remote_code=True,
                gpu_memory_utilization=self.gpu_memory_utilization,
                pre_tokenizer=self.tokenizer,
            )
            self.llm.setup()
        else:
            raise ValueError("Either base_url or hf_model must be provided.")
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TranslationDataGenerator":
        """
        Create a TranslationDataGenerator instance from a YAML configuration file.
        Args:
            yaml_path: Path to the YAML configuration file.
        Returns:
            TranslationDataGenerator instance.
        """
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        # Remove 'text' key if present, only pass valid constructor args
        config = dict(config)  # Make a copy
        config.pop('text', None)
        return cls(**config)

    def setup(self):
        self.text_splitter = TextSplitter(model_name=self.hf_model, hf_token=self.hf_token, max_token_per_chunk=self.max_token_per_chunk, tokenizer=self.tokenizer, spacy_model=self.spacy_model)
    
    def generate(self, llm_prompt: str | list[str], debug: bool = False) -> list[str]:
        """
        Main pipeline for translation generation.
        Args:
            llm_prompt: The input text to be translated (str or list of str).
        Returns:
            List of improved translations.
        """
        if isinstance(llm_prompt, str):
            chunks = self.text_splitter.split_long_text(llm_prompt, max_tokens=self.max_token_per_chunk)
        else:
            chunks = llm_prompt
        results = []
        for chunk in chunks:
            initial = self._run_init_translation(chunk)
            reflection = self._run_reflection(chunk, initial)
            improved = self._run_improve_translation(chunk, initial, reflection)
            extract_data = extract_content(improved, debug=debug)
            results.append(extract_data)
        return results

    def generate_from_yaml(self, yaml_path: str, text: str) -> str:
        """
        Run the translation pipeline using parameters from a YAML configuration file, with text provided in code.
        Args:
            yaml_path: Path to the YAML file containing parameters.
            text: The text to translate (provided in code).
        Returns:
            Concatenated translation string.
        """
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        # Only use parameters from YAML, ignore any text key
        translations = self.generate(text)
        return self.parse_response(translations)

    def parse_response(self, llm_response: str | list[str]) -> str:
        """
        Parse the LLM response(s) into a single formatted string.
        Args:
            llm_response: List of improved translations.
        Returns:
            Concatenated translation string.
        """
        if isinstance(llm_response, list):
            return "\n".join([x for x in llm_response if isinstance(x, str) and x is not None])
        return llm_response

    def _chat_llm(self, prompt: str) -> str:
        if self.base_url:
            response = self.openai_client.chat.completions.create(
                model=self.api_model,
                messages=[{"role": "user", "content": prompt}],
                stop=self.stop,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content
        else:
            chat_inputs = [
                [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": prompt}
                ]
            ]
            chat_outputs = self.llm.generate(chat_inputs)
            return chat_outputs[0]
    
    def _run_init_translation(self, text: str) -> str:
        # Run initial translation step
        prompt = INITIAL_TRANSLATION_PROMPT.format(
            source_lang=self.source_lang,
            source_text=text,
            target_lang=self.target_lang,
        )
        return self._chat_llm(prompt)

    def _run_reflection(self, text: str, initial_translation_result: str) -> str:
        # Run reflection step to improve translation
        if self.country:
            prompt = REFLECTION_COUNTRY_TRANSLATION_PROMPT.format(
                source_lang=self.source_lang,
                source_text=text,
                target_lang=self.target_lang,
                translation_1=initial_translation_result,
                country=self.country,
            )
        else:
            prompt = REFLECTION_TRANSLATION_PROMPT.format(
                source_lang=self.source_lang,
                source_text=text,
                target_lang=self.target_lang,
                translation_1=initial_translation_result,
            )
        return self._chat_llm(prompt)

    def _run_improve_translation(self, text: str, initial_translation_result: str, reflection_result: str) -> str:
        # Run improvement step to finalize translation
        prompt = IMPROVE_TRANSLATION_PROMPT.format(
            source_lang=self.source_lang,
            source_text=text,
            target_lang=self.target_lang,
            translation_1=initial_translation_result,
            reflection=reflection_result,
        )
        return self._chat_llm(prompt)

    def generate_from_dataframe(self, df, text_column: str, batch_size: int = 32, output_column: str = "translated_text", **kwargs) -> "pd.DataFrame":
        """
        Pipeline: Translate a pandas DataFrame column in batches and return DataFrame with new column.
        Args:
            df: pandas DataFrame containing text data.
            text_column: Name of the column to translate.
            batch_size: Number of rows per batch.
            output_column: Name of the column to store translations.
            kwargs: Extra arguments for self.generate.
        Returns:
            DataFrame with new column containing translations.
        """
        
        results = []
        n = len(df)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = df.iloc[start:end]
            texts = batch[text_column].tolist()
            batch_translations = self.generate(texts, **kwargs)
            results.extend(batch_translations)
        df[output_column] = results
        return df

    async def async_generate_from_dataframe(self, df, text_column: str, batch_size: int = 32, output_column: str = "translated_text", **kwargs):
        """
        Asynchronous pipeline: Translate a pandas DataFrame column in batches using async requests and return DataFrame with new column.
        Args:
            df: pandas DataFrame containing text data.
            text_column: Name of the column to translate.
            batch_size: Number of rows per batch.
            output_column: Name of the column to store translations.
            kwargs: Extra arguments for self.generate.
        Returns:
            DataFrame with new column containing translations.
        """
        import asyncio

        async def async_generate(texts, **kwargs):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.generate, texts, **kwargs)

        n = len(df)
        tasks = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = df.iloc[start:end]
            texts = batch[text_column].tolist()
            tasks.append(async_generate(texts, **kwargs))

        results = []
        batch_translations_list = await asyncio.gather(*tasks)
        for batch_translations in batch_translations_list:
            results.extend(batch_translations)
        df[output_column] = results
        return df

# Example usage for both function and YAML config
if __name__ == "__main__":
    text = "Once upon a time, there were three little pig brothers..."
    
    # generator = TranslationDataGenerator(
    #     base_url="http://127.0.0.1:8080/v1",                    # Base URL for local API (P.S: llama.cpp/ollama/vllm/TensorRT-LLM supports the OpenAI API format.)
    #     api_key="empty",                                        # API key (empty if not required)
    #     api_model="/home/user/.cache/llama.cpp/unsloth_gpt-oss-20b-GGUF_gpt-oss-20b-F16.gguf",    # API translation model
    #     hf_model="openai/gpt-oss-20b" ,                         # HuggingFace model for tokenization
    #     hf_token=None,                                          # HuggingFace authentication token
    #     temperature=1.0,                                        # Sampling temperature for generation
    #     top_p=1.0,                                              # Nucleus sampling parameter
    #     max_tokens=24576,                                       # Maximum tokens for output
    #     stop=["<|return|>","<|endoftext|>", "<|call|>"],        # Stop TOKEN sequences
    #     max_token_per_chunk=5000,                               # Max tokens per chunk for translation
    #     source_lang="English",                                  # Source language
    #     target_lang="Traditional Chinese",                      # Target language
    #     country="Taiwan",                                       # Country context for translation
    #     spacy_model="en_core_web_lg"                            # spaCy model for sentence splitting
    # )
    # generator.setup()
    # translations = generator.generate(text)
    # print("Generator Output:")
    # print(generator.parse_response(translations))

    generator = TranslationDataGenerator(
        hf_model="Qwen/Qwen3-4B-Instruct-2507" ,                # HuggingFace model for tokenization
        hf_token=None,                                          # HuggingFace authentication token
        temperature=1.0,                                        # Sampling temperature for generation
        top_p=1.0,                                              # Nucleus sampling parameter
        max_tokens=24576,                                       # Maximum tokens for output
        max_token_per_chunk=5000,                               # Max tokens per chunk for translation
        source_lang="English",                                  # Source language
        target_lang="Traditional Chinese",                      # Target language
        country="Taiwan",                                       # Country context for translation
        spacy_model="en_core_web_lg"                            # spaCy model for sentence splitting
    )
    generator.setup()
    translations = generator.generate(text)
    print("Generator vLLM Output:")
    print(generator.parse_response(translations))

    # YAML-based usage (parameters only, text provided in code)
    # generator_yaml = TranslationDataGenerator.from_yaml("config/translation_config.yaml")
    # generator_yaml.setup()
    # print(generator_yaml.generate_from_yaml("config/translation_config.yaml", text))

    # Pipeline DataFrame usage
    df = pd.DataFrame(
        {
            "text": [
                "Once upon a time, there were three little pig brothers...",
                "The quick brown fox jumps over the lazy dog.",
            ]
        }
    )
    df_translated = generator.generate_from_dataframe(df, text_column="text", batch_size=16)
    print(df_translated.head())

    # Async test for async_generate_from_dataframe
    import asyncio
    async def test_async_generate_from_dataframe():
        df_translated = await generator.async_generate_from_dataframe(df, text_column="text", batch_size=16)
        print("[Async]\n", df_translated.head())

    asyncio.run(test_async_generate_from_dataframe())