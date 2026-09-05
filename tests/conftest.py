import os
import sys

# Make the package dir importable so tests can `import unsloth_client` etc.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
