from .common import getLogger
from .nodes.LocalTranslator import LocalTranslatorNode

__all__ = [
    'NODE_CLASS_MAPPINGS',
    'NODE_DISPLAY_NAME_MAPPINGS',
    
]

__author__ = 'love2hina'
__email__ = 'webmaster@love2hina.net'
__version__ = '0.0.2'

NODE_CLASS_MAPPINGS = {
    'LocalTranslator': LocalTranslatorNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    'LocalTranslator': 'Local Translator',
}

getLogger().info('ComfyUI Local Translator loaded.')
