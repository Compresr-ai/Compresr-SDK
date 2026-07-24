"""First-party integrations for Compresr.

Each integration is an optional install:

    pip install compresr[langchain]
    pip install compresr[langgraph]
    pip install compresr[llamaindex]
    pip install compresr[litellm]

Importing a sub-package without its peer dependency raises a clear
``ImportError`` pointing at the right extra.
"""
