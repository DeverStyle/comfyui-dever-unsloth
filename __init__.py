import os
import sys

# ComfyUI loads this via spec_from_file_location on __init__.py, which breaks
# relative imports for a hyphenated folder name. Add the package dir to sys.path
# and use bare imports instead.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WEB_DIRECTORY = "./web"

try:
    from UnslothNodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:
    # UnslothNodes not present yet (early in development) — expose empty mappings
    # so the package still imports cleanly.
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
