"""Root conftest — prevent pytest from importing the plugin's __init__.py
as a standalone module (which breaks relative imports).

The plugin package is accessed during tests via the `plugins/` symlink
as `plugins.generative_ai_art`, where the relative imports work correctly.
"""

collect_ignore = ["__init__.py", "source.py"]
