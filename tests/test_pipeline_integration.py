"""
tests/test_pipeline_integration.py

End-to-end integration tests for:

    Sentence
        ↓
    SemanticExtractor
        ↓
    SemanticGraphBuilder
        ↓
    TargetSelector
        ↓
    EntityResolver

The tests verify:

    1. Explicit targets
    2. Valid within-sentence resolutions
    3. Correctly unresolved mentions

Sentiment is intentionally NOT tested yet.
"""

from __future__ import annotations

import pytest

from semantic.semantic_extractor import SemanticExtractor
from semantic.semantic_graph import SemanticGraphBuilder
from semantic.target_selector import TargetSelector
from semantic.entity_resolver import EntityResolver


# ============================================================
# Integration cases
# ============================================================

INTEGRATION_CASES = [
    {
        "name": "Microsoft acquisition",
        "sentence": (
            "Microsoft acquired Activision Blizzard "
            "in a major technology deal."
        ),
        "expected_targets": [
            "Microsoft",
            "Activision Blizzard",
        ],
        "expected_resolutions": {},
        "expected_unresolved": [],
    },

    {
        "name": "Nvidia revenue",
        "sentence": (
            "Nvidia reported record quarterly revenue "
            "after strong demand for AI chips."
        ),
        "expected_targets": [
            "Nvidia",
            "record quarterly revenue",
        ],
        "expected_resolutions": {},
        "expected_unresolved": [],
    },

    {
        "name": "Pfizer trial",
        "sentence": (
            "Pfizer reported positive results from "
            "its latest clinical trial."
        ),
        "expected_targets": [
            "Pfizer",
            "positive results from its latest clinical trial",
        ],
        "expected_resolutions": {},
        "expected_unresolved": [],
    },

    {
        "name": "Tesla company resolution",
        "sentence": (
            "Tesla shares rose after the company "
            "reported stronger sales."
        ),
        "expected_targets": [
            "Tesla shares",
            # "the company",
            "stronger sales",
        ],
        "expected_resolutions": {
            "the company": "Tesla",
        },
        "expected_unresolved": [],
    },

    {
        "name": "Apple product",
        "sentence": (
            "Apple announced a new iPhone model, "
            "which investors praised for its design."
        ),
        "expected_targets": [
            "Apple",
            "a new iPhone model",
            # "investors",
        ],
        "expected_resolutions": {},
        "expected_unresolved": [],
    },

    {
        "name": "Amazon cloud services",
        "sentence": (
            "Amazon cut prices on its cloud services "
            "as customers demanded lower costs."
        ),
        "expected_targets": [
            "Amazon",
            "prices",
            "customers",
        ],
        "expected_resolutions": {},
        "expected_unresolved": [],
    },

    {
        "name": "Federal Reserve",
        "sentence": (
            "The Federal Reserve kept interest rates "
            "unchanged on Wednesday."
        ),
        "expected_targets": [
            "The Federal Reserve",
            "interest rates",
        ],
        "expected_resolutions": {},
        "expected_unresolved": [],
    },

    {
        "name": "France and Google",
        "sentence": (
            "France imposed new sanctions on Google "
            "after the company violated the rules."
        ),
        "expected_targets": [
            "France",
            "Google",
        ],
        "expected_resolutions": {
            "the company": "Google",
        },
        "expected_unresolved": [],
    },

    {
        "name": "Tim Cook and Apple",
        "sentence": (
            "Tim Cook said Apple would increase "
            "investment in India."
        ),
        "expected_targets": [
            "Tim Cook",
            "Apple",
            "investment in India",
        ],
        "expected_resolutions": {},
        "expected_unresolved": [],
    },

    {
        "name": "Unresolved generic references",
        "sentence": (
            "The company launched a new product "
            "after its competitors lowered prices."
        ),
        "expected_targets": [
            "a new product",
            "prices",
            "its competitors",
        ],
        "expected_resolutions": {
            "the company": None,
        },
        "expected_unresolved": [
            "the company",
            "its",
        ],
    },
]


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture(scope="module")
def extractor():
    return SemanticExtractor()


@pytest.fixture(scope="module")
def graph_builder():
    return SemanticGraphBuilder()


@pytest.fixture(scope="module")
def selector():
    return TargetSelector()


@pytest.fixture(scope="module")
def resolver():
    return EntityResolver()


# ============================================================
# Pipeline
# ============================================================


def run_pipeline(
    sentence,
    extractor,
    graph_builder,
    selector,
    resolver,
):
    """
    Run the actual production pipeline.
    """

    representation = extractor.extract(
        sentence
    )

    graph = graph_builder.build(
        representation
    )

    targets = selector.select_all(
        graph
    )

    resolution_result = resolver.resolve(
        [graph],
        [targets],
    )

    return (
        representation,
        graph,
        targets,
        resolution_result,
    )


# ============================================================
# Target helpers
# ============================================================


def _target_text(target):
    """
    TargetSelection uses `.target` for the actual target text.
    """

    if isinstance(target, str):
        return target

    if hasattr(target, "target"):
        return target.target

    if isinstance(target, dict):

        if "target" in target:
            return target["target"]

        if "text" in target:
            return target["text"]

    raise AssertionError(
        f"Could not determine target text from: {target!r}"
    )


def _target_texts(targets):
    return [
        _target_text(target)
        for target in targets
    ]


# ============================================================
# Resolution helpers
# ============================================================


def _resolved_entities(
    resolution_result,
):
    return getattr(
        resolution_result,
        "resolved_entities",
        [],
    )


def _unresolved_mentions(
    resolution_result,
):
    return getattr(
        resolution_result,
        "unresolved_mentions",
        [],
    )


def _canonical_name(entity):
    """
    ResolvedEntity uses `.canonical_name`.
    """

    if hasattr(entity, "canonical_name"):
        return entity.canonical_name

    if hasattr(entity, "text"):
        return entity.text

    if isinstance(entity, dict):
        return (
            entity.get("canonical_name")
            or entity.get("text")
            or entity.get("canonical")
        )

    return str(entity)


def _mention_text(mention):
    """
    Mention uses `.text`.
    """

    if isinstance(mention, str):
        return mention

    if hasattr(mention, "text"):
        return mention.text

    if isinstance(mention, dict):
        return (
            mention.get("text")
            or mention.get("mention")
        )

    return str(mention)


def _resolution_pairs(
    resolution_result,
):
    """
    Convert ResolutionResult into:

        mention -> canonical entity

    Example:

        {
            "tesla": "Tesla",
            "the company": "Tesla"
        }
    """

    pairs = {}

    for entity in _resolved_entities(
        resolution_result
    ):

        canonical = _canonical_name(
            entity
        )

        mentions = getattr(
            entity,
            "mentions",
            [],
        )

        if isinstance(entity, dict):
            mentions = entity.get(
                "mentions",
                mentions,
            )

        for mention in mentions:

            text = _mention_text(
                mention
            )

            if text:
                pairs[
                    text.lower()
                ] = canonical

    return pairs


def _unresolved_texts(
    resolution_result,
):
    """
    Return all unresolved mention texts.
    """

    return {
        _mention_text(
            mention
        ).lower()
        for mention in _unresolved_mentions(
            resolution_result
        )
    }


# ============================================================
# 1. Explicit target tests
# ============================================================


@pytest.mark.parametrize(
    "case",
    INTEGRATION_CASES,
    ids=lambda case: case["name"],
)
def test_explicit_targets(
    case,
    extractor,
    graph_builder,
    selector,
    resolver,
):
    """
    Every expected target must be selected by TargetSelector.
    """

    (
        _representation,
        _graph,
        targets,
        _resolution_result,
    ) = run_pipeline(
        case["sentence"],
        extractor,
        graph_builder,
        selector,
        resolver,
    )

    actual_targets = _target_texts(
        targets
    )

    for expected in case[
        "expected_targets"
    ]:

        assert expected in actual_targets, (
            f"\n"
            f"Sentence: {case['sentence']!r}\n"
            f"Expected target: {expected!r}\n"
            f"Actual targets: {actual_targets!r}"
        )


# ============================================================
# 2. Valid resolution tests
# ============================================================


@pytest.mark.parametrize(
    "case",
    [
        case
        for case in INTEGRATION_CASES
        if any(
            value is not None
            for value in case[
                "expected_resolutions"
            ].values()
        )
    ],
    ids=lambda case: case["name"],
)
def test_valid_resolutions(
    case,
    extractor,
    graph_builder,
    selector,
    resolver,
):
    """
    Verify references that should resolve.
    """

    (
        _representation,
        _graph,
        _targets,
        resolution_result,
    ) = run_pipeline(
        case["sentence"],
        extractor,
        graph_builder,
        selector,
        resolver,
    )

    pairs = _resolution_pairs(
        resolution_result
    )

    unresolved = _unresolved_texts(
        resolution_result
    )

    for mention, expected_entity in case[
        "expected_resolutions"
    ].items():

        if expected_entity is None:
            continue

        normalized = mention.lower()

        assert normalized not in unresolved, (
            f"\n"
            f"Expected {mention!r} to resolve "
            f"to {expected_entity!r}.\n"
            f"Sentence: {case['sentence']!r}\n"
            f"Unresolved: {sorted(unresolved)!r}"
        )

        actual_entity = pairs.get(
            normalized
        )

        assert actual_entity == expected_entity, (
            f"\n"
            f"Sentence: {case['sentence']!r}\n"
            f"Mention: {mention!r}\n"
            f"Expected entity: {expected_entity!r}\n"
            f"Actual entity: {actual_entity!r}\n"
            f"Resolution pairs: {pairs!r}"
        )


# ============================================================
# 3. Unresolved mention tests
# ============================================================


@pytest.mark.parametrize(
    "case",
    [
        case
        for case in INTEGRATION_CASES
        if case["expected_unresolved"]
    ],
    ids=lambda case: case["name"],
)
def test_unresolved_mentions(
    case,
    extractor,
    graph_builder,
    selector,
    resolver,
):
    """
    Verify that genuinely unresolved references remain
    unresolved.

    Important:

        "its competitors"

    is currently represented by EntityResolver V2 as the
    unresolved pronoun "its".

    Therefore the test expects "its", matching the current
    resolver contract.
    """

    (
        _representation,
        _graph,
        _targets,
        resolution_result,
    ) = run_pipeline(
        case["sentence"],
        extractor,
        graph_builder,
        selector,
        resolver,
    )

    actual_unresolved = _unresolved_texts(
        resolution_result
    )

    for expected in case[
        "expected_unresolved"
    ]:

        assert expected.lower() in actual_unresolved, (
            f"\n"
            f"Expected {expected!r} to remain unresolved.\n"
            f"Sentence: {case['sentence']!r}\n"
            f"Actual unresolved mentions: "
            f"{sorted(actual_unresolved)!r}"
        )


# ============================================================
# 4. No invented coreference
# ============================================================


def test_unresolved_case_does_not_invent_entities(
    extractor,
    graph_builder,
    selector,
    resolver,
):
    """
    The resolver must not invent an antecedent for a generic
    reference when no suitable antecedent exists.
    """

    sentence = (
        "The company launched a new product "
        "after its competitors lowered prices."
    )

    (
        _representation,
        _graph,
        _targets,
        resolution_result,
    ) = run_pipeline(
        sentence,
        extractor,
        graph_builder,
        selector,
        resolver,
    )

    pairs = _resolution_pairs(
        resolution_result
    )

    assert pairs.get(
        "the company"
    ) is None, (
        "Resolver invented an antecedent for "
        "'the company'."
    )

    assert pairs.get(
        "its"
    ) is None, (
        "Resolver invented an antecedent for "
        "'its'."
    )


# ============================================================
# 5. Focused positive resolution tests
# ============================================================


def test_tesla_company_resolution(
    extractor,
    graph_builder,
    selector,
    resolver,
):
    """
    Expected:

        the company -> Tesla
    """

    sentence = (
        "Tesla shares rose after the company "
        "reported stronger sales."
    )

    (
        _representation,
        _graph,
        _targets,
        resolution_result,
    ) = run_pipeline(
        sentence,
        extractor,
        graph_builder,
        selector,
        resolver,
    )

    pairs = _resolution_pairs(
        resolution_result
    )

    assert pairs.get(
        "the company"
    ) == "Tesla", (
        f"Expected 'the company' -> 'Tesla'. "
        f"Got: {pairs!r}"
    )


def test_google_company_resolution(
    extractor,
    graph_builder,
    selector,
    resolver,
):
    """
    Expected:

        the company -> Google
    """

    sentence = (
        "France imposed new sanctions on Google "
        "after the company violated the rules."
    )

    (
        _representation,
        _graph,
        _targets,
        resolution_result,
    ) = run_pipeline(
        sentence,
        extractor,
        graph_builder,
        selector,
        resolver,
    )

    pairs = _resolution_pairs(
        resolution_result
    )

    assert pairs.get(
        "the company"
    ) == "Google", (
        f"Expected 'the company' -> 'Google'. "
        f"Got: {pairs!r}"
    )


# ============================================================
# 6. Full pipeline smoke test
# ============================================================


def test_all_integration_cases_execute(
    extractor,
    graph_builder,
    selector,
    resolver,
):
    """
    Every sentence must execute through all four pipeline
    stages without raising an exception.
    """

    for case in INTEGRATION_CASES:

        (
            representation,
            graph,
            targets,
            resolution_result,
        ) = run_pipeline(
            case["sentence"],
            extractor,
            graph_builder,
            selector,
            resolver,
        )

        assert representation is not None
        assert graph is not None
        assert targets is not None
        assert resolution_result is not None

