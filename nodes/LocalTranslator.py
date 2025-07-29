import logging
from textwrap import dedent
from typing import (
    Any,
    Literal,
    Optional,
)

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextGenerationPipeline,
)

from comfy import model_management

from ..common import MODEL_PATH
from .ProxyForLM import ProxyForLM

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

proxy: Optional[ProxyForLM] = None
tokenizer: Optional[Any] = None

def load_model() -> tuple[AutoModelForCausalLM, Any]:
    global proxy
    global tokenizer

    if proxy is None:
        proxy = ProxyForLM.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
        )
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
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

    TRANSLATE_PLACEHOLDER: str = '%TRANSLATE%'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'string': ('STRING', { 'multiline': True }),
            },
            'optional': {
                'optional': ('STRING', { 'multiline': True, 'default': '' }),
                'max_tokens': ('INT', { 'default': 512, 'min': 1, 'max': 2048, 'step': 1 }),
            },
        }
    
    @classmethod
    def VALIDATE_INPUTS(cls, string: str, optional: str, max_tokens: int) -> Literal[True] | str:
        if not string:
            logger.warning("[Local Translator] Input string is empty.")

        if optional:
            # 文字列中に %TRANSLATE% が含まれていない場合
            if optional.find(LocalTranslatorNode.TRANSLATE_PLACEHOLDER) == -1:
                return f"optional string must contain '{LocalTranslatorNode.TRANSLATE_PLACEHOLDER}' to indicate where the translated text should be inserted."
            # 文字列中に %TRANSLATE% が複数含まれている場合
            if optional.count(LocalTranslatorNode.TRANSLATE_PLACEHOLDER) > 1:
                return f"optional string must contain '{LocalTranslatorNode.TRANSLATE_PLACEHOLDER}' only once to indicate where the translated text should be inserted."

        return True
    
    def translate(self, string: str, optional: str = '', max_tokens: int = 512) -> tuple[str]:
        logger.debug(f"[Local Translator] Input text: {string}, max_tokens: {max_tokens}")
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
                    Translated: He has been crazy about "THE IDOLM@STER" for about 20 years.

                    As you can see from these examples, there is an intention to display trademarks and other information precisely, so MUST use the specified keywords as is.
                ''')
            },
            { 'role': 'user', 'content': string },
        ]
        params = {
            'max_new_tokens': max_tokens,
            'do_sample': False,
            'num_beams': 4,
        }
        response = pipeline(prompts, **params)
        translated = response[0]['generated_text'][-1]['content']
        logger.debug(f"[Local Translator] Translated text: {translated}")

        if optional:
            # %TRANSLATE% を置換
            result = optional.replace(LocalTranslatorNode.TRANSLATE_PLACEHOLDER, translated, 1)
        else:
            result = translated
        
        return (result,)
    
    RETURN_TYPES = ('STRING',)
    FUNCTION = translate.__name__
    CATEGORY = 'Text Nodes'
