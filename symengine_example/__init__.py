from pathlib import Path

CURRENT_DIR = Path(__file__).parent.parent
_ENABLE_SYMENGINE = False


def enable_symengine():
    global _ENABLE_SYMENGINE
    _ENABLE_SYMENGINE = True
