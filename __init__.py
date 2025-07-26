import logging

from .nodes.LocalTranslator import LocalTranslatorNode

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# WEB_DIRECTORY = './js'
NODE_CLASS_MAPPINGS = {
    'LocalTranslator': LocalTranslatorNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    'LocalTranslator': 'Local Translator',
}

logger.info('[Local Translator] ComfyUI Local Translator loaded.')
