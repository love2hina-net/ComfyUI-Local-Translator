import logging

from .nodes.LocalTranslator import LocalTranslatorNode

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

__all__ = [
    'NODE_CLASS_MAPPINGS',
    'NODE_DISPLAY_NAME_MAPPINGS',
    
]

__author__ = 'love2hina'
__email__ = 'webmaster@love2hina.net'
__version__ = '0.0.1'

NODE_CLASS_MAPPINGS = {
    'LocalTranslator': LocalTranslatorNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    'LocalTranslator': 'Local Translator',
}

logger.info('[Local Translator] ComfyUI Local Translator loaded.')
