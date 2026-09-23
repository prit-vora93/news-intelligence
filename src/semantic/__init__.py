"""
semantic package
-----------------

Convention (important — follow this everywhere):

    Always import using the package-qualified path:

        from semantic.semantic_extractor import SemanticExtractor
        from semantic.semantic_graph import SemanticGraphBuilder
        from semantic.target_selector import TargetSelector

    Always RUN scripts as modules from the project root (one level
    above this semantic/ folder), e.g.:

        python -m semantic.test_target_selector
        python -m semantic.test_semantic_extractor

    Do NOT run them as `python semantic/test_target_selector.py`
    directly — that breaks the package-relative imports.
"""