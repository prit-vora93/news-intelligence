"""
quick_judge.py
--------------

Quick eyeball check: run a batch of sentences through the full
pipeline (extractor -> graph -> select_all) and print a compact
one-line-per-target summary, so you can scan results fast without
digging through test_target_selector.py's full verbose output.

This is NOT a replacement for test_target_selector.py — just a
faster way to sanity-check before we update that file properly.
"""

from semantic.semantic_extractor import SemanticExtractor
from semantic.semantic_graph import SemanticGraphBuilder
from semantic.target_selector import TargetSelector


SENTENCES = [
    # Coordination cases (the pattern we just fixed)
    "Google announced layoffs and Meta announced hiring.",
    "The United States and the European Union imposed new sanctions on Russia.",
    "Apple and Google are competing to develop new AI products.",

    # Original single-target-style sentences
    "Microsoft acquired Activision Blizzard.",
    "President Trump announced a new economic policy today.",
    "Elon Musk announced a new plan for Tesla.",
    "Nvidia reported record quarterly revenue.",
    "OpenAI announced a new partnership with Microsoft.",
    "Google introduced several new artificial intelligence products.",
    "Tesla shares rose after the company reported stronger sales.",
    "The Federal Reserve kept interest rates unchanged.",
    "China's President Xi Jinping met with European Union officials.",
    "Meta announced that it will invest billions of dollars in artificial intelligence.",
    "Amazon founder Jeff Bezos met with Indian Prime Minister Narendra Modi.",
    "Apple launched a new iPhone model.",
    "Google announced plans to cut thousands of jobs worldwide.",
    "Oil prices climbed after Saudi Arabia announced additional production cuts.",
    "Investors sold technology stocks after the Federal Reserve raised interest rates.",
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
                    f"      -> {t.target!r:25} "
                    f"role={t.role:8} "
                    f"conf={t.confidence}"
                )

        print()


if __name__ == "__main__":
    main()
