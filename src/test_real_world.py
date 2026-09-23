"""
src/test_real_world.py
----------------------

Real-world test for the candidate ranking system.

IMPORTANT:
This test does NOT provide a gold target to the model.

We want to see whether the existing candidate generator +
candidate ranker can make useful predictions on real-world
news-style sentences.
"""

import sys
from pathlib import Path

# ------------------------------------------------------------
# Project root
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ------------------------------------------------------------
# Import existing ranking pipeline
# ------------------------------------------------------------

from src.inspect_candidate_ranking import (
    rank_candidates,
)


# ============================================================
# Real-world test sentences
# ============================================================

TEST_SENTENCES = [

    # --------------------------------------------------------
    # PERSON
    # --------------------------------------------------------

    "President Trump announced a new economic policy today.",

    "Donald Trump met with European leaders in Washington.",

    "Microsoft CEO Satya Nadella spoke at the conference.",

    "Elon Musk announced a new plan for Tesla.",

    "Apple CEO Tim Cook visited India this week.",

    # --------------------------------------------------------
    # ORGANIZATION
    # --------------------------------------------------------

    "OpenAI announced a new partnership with Microsoft.",

    "Nvidia reported record quarterly revenue after strong demand for AI chips.",

    "Google introduced several new artificial intelligence products.",

    "Tesla shares rose after the company reported stronger sales.",

    # --------------------------------------------------------
    # LOCATION
    # --------------------------------------------------------

    "The company announced a new factory in Gujarat.",

    "Officials in Washington discussed the proposed agreement.",

    # --------------------------------------------------------
    # MULTIPLE ENTITIES
    # --------------------------------------------------------

    "Microsoft acquired Activision Blizzard in a major technology deal.",

    "Amazon founder Jeff Bezos met with Indian Prime Minister Narendra Modi.",

    "Apple and Google are competing to develop new AI products.",

    # --------------------------------------------------------
    # More realistic news sentences
    # --------------------------------------------------------

    "The Federal Reserve kept interest rates unchanged on Wednesday.",

    "Oil prices climbed after Saudi Arabia announced additional production cuts.",

    "The United States imposed new sanctions on Russia.",

    "China's President Xi Jinping met with European Union officials.",

    "Meta announced that it will invest billions of dollars in artificial intelligence.",

]


# ============================================================
# Display helper
# ============================================================

def print_separator():
    print("-" * 70)


# ============================================================
# Run tests
# ============================================================

def run_tests():

    print()
    print("=" * 70)
    print("REAL-WORLD CANDIDATE RANKING TEST")
    print("=" * 70)

    print()
    print("IMPORTANT:")
    print("No gold targets are provided.")
    print("The system must decide what is important.")
    print()

    for index, sentence in enumerate(
        TEST_SENTENCES,
        start=1,
    ):

        print()
        print("=" * 70)
        print(f"Example {index}")
        print("=" * 70)

        print()
        print("Sentence:")
        print(sentence)

        print()
        print("Ranked candidates:")
        print("-" * 70)

        try:

            # Generate candidates using the existing pipeline
            # candidates = generate_candidates(sentence)

            # Rank candidates using the trained ranker
            ranked = rank_candidates(
                sentence,
                candidates
            )

        except Exception as exc:

            print()
            print("ERROR:")
            print(exc)

            continue

        if not ranked:

            print("No candidates found.")
            continue

        # ----------------------------------------------------
        # Display top candidates
        # ----------------------------------------------------

        for rank, candidate in enumerate(
            ranked[:10],
            start=1,
        ):

            # ------------------------------------------------
            # Handle different result formats safely
            # ------------------------------------------------

            if isinstance(candidate, dict):

                text = candidate.get(
                    "candidate",
                    candidate.get(
                        "text",
                        "",
                    ),
                )

                score = candidate.get(
                    "score",
                    candidate.get(
                        "probability",
                        None,
                    ),
                )

                source = candidate.get(
                    "source",
                    "",
                )

                ner_label = candidate.get(
                    "ner_label",
                    None,
                )

            else:

                text = str(candidate)
                score = None
                source = ""
                ner_label = None

            if score is None:

                print(
                    f"{rank:2d}. "
                    f"{text!r:45} "
                    f"{source:25} "
                    f"NER={ner_label}"
                )

            else:

                print(
                    f"{rank:2d}. "
                    f"{text!r:45} "
                    f"score={score:.4f} "
                    f"{source:25} "
                    f"NER={ner_label}"
                )

        print()

        print("Top prediction:")

        first = ranked[0]

        if isinstance(first, dict):

            top_text = first.get(
                "candidate",
                first.get(
                    "text",
                    "",
                ),
            )

        else:

            top_text = str(first)

        print(f"  {top_text!r}")

        print()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    run_tests()