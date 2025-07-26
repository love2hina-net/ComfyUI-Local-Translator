import logging
from typing import Optional
from pathlib import Path

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextGenerationPipeline
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

pipeline: Optional[TextGenerationPipeline] = None

def load_pipeline() -> TextGenerationPipeline:
    global pipeline
    if pipeline is None:
        path = (Path(__file__).parent / '..' / 'models' / 'phi-4-unsloth-bnb-4bit').resolve()
        model = AutoModelForCausalLM.from_pretrained(
            path,
            local_files_only=True,
            load_in_4bit=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            local_files_only=True,
            load_in_4bit=True,
        )
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
        _pipeline = load_pipeline()

        prompts = [
            { 'role': 'system', 'content': 'You are a translator. Please translate the message from the user into English.' },
            { 'role': 'user', 'content': string },
        ]
        response = _pipeline(prompts, max_new_tokens=max_tokens)
        translated = response[0]['generated_text'][-1]['content']
        logger.debug(f'[Local Translator] Translated text: {translated}')
        
        return (translated,)
    
    RETURN_TYPES = ('STRING',)
    FUNCTION = translate.__name__
    CATEGORY = 'Text Nodes'
