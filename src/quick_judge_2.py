"""
quick_judge_2.py
-----------------

Second batch of real-world sentences, deliberately different from
the first quick_judge.py set. Covers patterns we haven't stressed
yet: passive voice, negation, three-way coordination, non-tech
domains (sports, health, climate), and short/simple sentences.
"""

from semantic.semantic_extractor import SemanticExtractor
from semantic.semantic_graph import SemanticGraphBuilder
from semantic.target_selector import TargetSelector


SENTENCES = [
    # Passive voice
    "Twitter was acquired by Elon Musk in 2022.",
    "Thousands of workers were laid off by Amazon last quarter.",

    # Negation
    "The Federal Reserve did not raise interest rates this month.",
    "Apple has not confirmed plans for a foldable iPhone.",

    # Three-way coordination
    "Amazon, Google, and Microsoft are all investing in AI infrastructure.",
    "The UK, France, and Germany agreed on a joint defense plan.",

    # Non-tech domains
    "Manchester United defeated Liverpool in a thrilling match.",
    "The World Health Organization declared the outbreak a global emergency.",
    "Scientists warned that rising sea levels threaten coastal cities.",

    # Short/simple sentences
    "Netflix raised its subscription prices.",
    "Boeing delayed the delivery of its new aircraft.",

    # Comparative / contrast structure
    "While Apple's revenue grew, Samsung's profits declined sharply.",

    # Quote-attribution style (common in news)
    "The company said it expects strong growth next year.",

    # Numbers-heavy financial sentence
    "Tesla's stock fell 8% after missing delivery targets.",
]


def main() -> None:

    print("Loading pipeline...")
    extractor = SemanticExtractor()
    builder = SemanticGraphBuilder()
    selector = TargetSelector()
    print("Ready.\n")

    for index, sentence in enumerate(SENTENCES, start=1):

        rep = extractor.extract(sentence)
        graph = builder.build(rep)
        targets = selector.select_all(graph)

        print(f"[{index:2}] {sentence}")

        if not targets:
            print("      -> NO TARGETS FOUND")
        else:
            for t in targets:
                print(
                    f"      -> {t.target!r:35} "
                    f"role={t.role:8} "
                    f"conf={t.confidence}"
                )

        print()


if __name__ == "__main__":
    main()
