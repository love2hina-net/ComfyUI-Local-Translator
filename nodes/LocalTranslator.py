import logging
from pathlib import Path
from textwrap import dedent
from typing import (
    Any,
    Optional,
)

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    TextGenerationPipeline,
)

from comfy import model_management

from .ProxyForLM import ProxyForLM

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

MODEL_PATH = (Path(__file__).parent / '..' / 'models' / 'phi-4-unsloth-bnb-4bit').resolve()

proxy: Optional[ProxyForLM] = None
tokenizer: Optional[Any] = None

class ProxyLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self):
        pass

def load_model() -> tuple[AutoModelForCausalLM, Any]:
    global proxy
    global tokenizer

    if proxy is None:
        proxy = ProxyForLM.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            load_in_4bit=True,
        )
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            load_in_4bit=True,
        )
    
    model_management.load_models_gpu([proxy])
    return proxy.model, tokenizer

def build_pipeline() -> TextGenerationPipeline:
    model, tokenizer = load_model()
    pipeline = transformers.pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
    )
    return pipeline

class LocalTranslatorNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'string': ('STRING', { "multiline": True }),
            },
            'optional': {
                'max_tokens': ('INT', { "default": 512, "min": 1, "max": 2048, "step": 1 }),
            },
        }
    
    def translate(self, string: str, max_tokens: int = 512) -> tuple[str]:
        logger.debug(f'[Local Translator] Input text: {string}, max_tokens: {max_tokens}')
        pipeline = build_pipeline()

        prompts = [
            { 'role': 'system', 'content': dedent('''
                You are a translator. Please translate the message from the user into English.
                                                  
                The special rules are as follows:

                1. _SENTENCE_[_KEYWORD_]

                    Make sure the KEYWORD in the brackets expresses the previous SENTENCE.

                    Example:
                    * Input: 大きなツインテール[pigtails hair]が彼女の特徴だ。<br/>
                    Translated: Her big pigtails hair are her defining feature.

                2. [_SENTENCE_|_KEYWORD_]

                    The SENTENCE before the vertical bars in square brackets must be expressed with the KEYWORD after the vertical bar.

                    Example:
                    * Input: 彼女は大きな赤い[ランドセル|randoseru]を背負って学校に向かった。<br/>
                    Translated: She headed to school with a big red randoseru on her back.
                    * Input: 彼は約20年にわたり[アイドルマスター|"THE IDOLM@STER"]に夢中だ。<br/>
                    Translated: He has been crazy fun about "THE IDOLM@STER" for a long time about 20 years.

                    As you can see from these examples, there is an intention to display trademarks and other information precisely, so MUST use the specified keywords as is.
                ''')
            },
            { 'role': 'user', 'content': string },
        ]
        response = pipeline(prompts, max_new_tokens=max_tokens)
        translated = response[0]['generated_text'][-1]['content']
        logger.debug(f'[Local Translator] Translated text: {translated}')
        
        return (translated,)
    
    RETURN_TYPES = ('STRING',)
    FUNCTION = translate.__name__
    CATEGORY = 'Text Nodes'
