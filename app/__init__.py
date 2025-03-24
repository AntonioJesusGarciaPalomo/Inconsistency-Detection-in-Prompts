"""
Inconsistency Detection API application package.
"""

import logging

# Configure basic logging - will be overridden by main.py configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

# Package version
__version__ = '1.0.0'