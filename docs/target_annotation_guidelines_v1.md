# Target Annotation Guidelines (Draft v1)

Purpose: define what counts as a valid sentiment target BEFORE building
the labeled dataset (pipeline step 7), so labeling is consistent.
This directly resolves open questions raised while testing
`target_selector.py`.

---

## 1. What counts as a valid target?

A valid target is a **named entity or specific noun phrase that a
reader could hold a sentiment/opinion toward** in the context of the
sentence: a company, person, country, organization, product, or
policy/action.

**Not valid targets** (reject these, even if the selector surfaces them):
- Bare pronouns ("it", "they", "he") — not resolvable without
  coreference; excluded upstream already.
- Purely generic phrases ("the company", "officials", "investors")
  — already filtered by `GENERIC_TERMS`.
- Weak entity types: dates, times, percentages, cardinal numbers
  ("quarter", "2022", "8%") — already filtered by `WEAK_ENTITY_LABELS`.

## 2. RESOLVED — full phrase + tagged embedded entity

Example: `"Elon Musk announced a new plan for Tesla."` selects
`'a new plan for Tesla'` as the object target. The target TEXT stays
as the full phrase, but the target now also carries an
`embedded_entities` list — any recognized named entities found
inside that phrase (here: `['Tesla']`).

Decision: **(C) Keep both.** Full phrase preserved as the target
text (no information lost), embedded named entities tagged
separately (gives dataset-building and future ML steps a cheap way
to recover the short-entity view without re-parsing). Implemented in
`target_selector.py` — see `TargetSelection.embedded_entities`.

## 3. ADDENDUM — long-clause targets fixed (confirmed via labeling trial)

The tension flagged when Section 2 was resolved ("full phrases can
get too long") stopped being theoretical once real labeling was
attempted: a 32-word target ("any doubts based on the judge's
opinions...that Judge Gorsuch would be a reliable conservative...")
showed up in an actual 33-record manual labeling trial, confirming
the concern was real, not hypothetical.

**Fix 1 (primary) — clause-boundary exclusion.** The dominant cause
was relative clauses (`relcl`) and clausal complements/modifiers
(`acl`, `ccomp`) getting absorbed wholesale into a candidate's "own
span" text in `semantic_extractor.py`'s `_own_span_tokens()`. Added
these dep types to the existing exclusion list (alongside
conj/cc/punct/entity-like-appos). Confirmed fix: the 32-word case
became `"any doubts"` (2 words); another case
("ways in which we can recreate a common space...") became just
`"ways"`.

**Fix 2 (safety net) — MAX_TARGET_WORDS length cap.** After Fix 1,
a smaller residual tail remained (up to 40 words) caused by a
DIFFERENT structure: chains of stacked prepositional phrases
("the possibility of X by Y under Z...") and clauses buried inside
`xcomp`/infinitival constructions that Fix 1 doesn't cover. Added
`TargetSelector.MAX_TARGET_WORDS = 15` and `_apply_length_cap()`:
if a candidate exceeds 15 words, shorten to its first embedded
entity (if one exists) or drop the candidate entirely (if none
does) -- deliberately NOT keep a long, entity-less phrase just to
avoid losing a candidate. Applied consistently across select_all()'s
object/subject loops AND the noun-phrase fallback path. Confirmed:
max target length across a 500-sentence real-data run dropped from
40 words to exactly 15 (the cap boundary), with the full sentence
always preserved alongside target_text in dataset records so no
context is actually lost from the labeling process -- only from the
target's own identifying text.

## 4. Confidence tiers — what they mean for labeling

- `high` — event's primary semantic role (e.g. object of an
  object-focused verb like "acquire"). Label these first when
  building the dataset; highest signal-to-noise.
- `medium` — secondary role of the same event, OR any role of an
  unclassified verb. Still worth labeling, but expect more borderline
  cases.
- `low` — fallback candidate (no event at all; entity or noun-phrase
  scan). Use sparingly — these sentences often don't have a clear
  actionable target at all (e.g. "Uncertainty in global markets.").
- `none` — no target found. These sentences should probably be
  excluded from the dataset rather than force-labeled.

## 5. Multi-target sentences

A single sentence CAN and SHOULD produce multiple independent
targets when applicable (per NewsMTSC's design). Confirmed working
patterns:
- Coordinated entities: "Amazon, Google, and Microsoft are investing"
  → 3 separate targets.
- Multiple events: "Google announced layoffs and Meta announced
  hiring" → Google, Meta, hiring as separate targets.

When labeling, each target in the same sentence gets its OWN
sentiment label — they are not assumed to share the same sentiment.

## 6. Pronouns and unresolved references

Sentences like `"Meta announced that it will invest billions..."`
currently only surface `Meta` and `billions of dollars` — the `it`
(which really refers to Meta) is dropped, not resolved.

**For now:** accept this gap. Do NOT hand-label "it" as a target.
This is explicitly deferred to `entity_resolver.py` (next component).
Once that exists, sentences with unresolved pronouns can be
re-processed and re-labeled.

## 7. Known systematic gaps to keep in mind while labeling

- Passive-voice sentences correctly extract the real actor now
  (e.g. "Elon Musk" from "...was acquired by Elon Musk"), but the
  semantic **role** label may not reflect true semantic
  agency (the passive subject is labeled "subject" even though
  it's semantically the one being acted upon). Don't rely on the
  `role` field alone as a sentiment-polarity cue.
- Verb classification (`OBJECT_FOCUSED_EVENTS` / `SUBJECT_FOCUSED_EVENTS`)
  is a ~30-verb hardcoded list. Sentences with unlisted verbs still
  produce candidates (at `medium` confidence for both roles), just
  without the confidence boost. Don't assume `medium` confidence
  means "unlikely to be a real target" — many are simply using a
  verb we haven't classified yet.
- `en_core_web_sm`'s NER occasionally mistags entity types (e.g.
  tagged "Activision Blizzard" as PERSON, not ORG). Entity TEXT
  matching still works fine; don't rely on entity LABEL for anything
  label-critical without spot-checking.

## 8. Status update (post entity_resolver.py + evaluation session)

`entity_resolver.py` was built, scoped (see
`entity_resolver_scope_v1.md`), and tested — including a Category A
fix for embedded pronouns/generic terms. The pipeline was then
evaluated against REAL gold data (`data/processed/train_target.csv`,
9,316 rows) via the new `evaluate_target_selector.py`, going from
71.3% -> 96.4% recall across four targeted fixes (copular verbs,
dative objects, prepositional-phrase traversal, plus the
GENERIC_TERMS gap this doc's Section 2 decision surfaced).

Full fix-by-fix history is in `entity_resolver_scope_v1.md` Section
8. The "Facebook" coordination/merged-entity-span bug mentioned in
an earlier draft of this section has since been FIXED (see that
doc's Section 5) — `_is_entity_match()` now falls back to substring
matching for cases where spaCy merges multiple real entities into
one malformed span.

**Since then:** moved into pipeline step 7 (dataset generation) via
`build_dataset.py`, which runs the full pipeline over real sentences
from `data/processed/train_target.csv` and auto-fills sentiment
labels from existing gold data wherever a trustworthy match exists
(`gold_exact` / `gold_embedded_or_resolved`), leaving only genuinely
new candidates as `needs_annotation`. A 33-record manual labeling
trial surfaced three more real bugs, all now fixed:
- `"what"` leaking through as a bare target (same class as the
  earlier who/which fix, just a missed word)
- possessive spacing artifacts ("Hillary 's") -- fixed at the
  SOURCE this time (`_subtree_text()`'s token-joining logic), not
  patched downstream, so it holds across every code path
- long-clause targets (up to 40 words) -- fixed via clause-boundary
  exclusion (relcl/acl/ccomp) plus a 15-word safety-net cap (see
  Section 3's addendum above)

**Current state:** `dataset_v1.jsonl` (500 sentences, ~1,560
records) is clean across every check run so far -- no relative
pronoun leaks, no spacing artifacts, target length capped at 15
words, max seen. ~21% of records auto-labeled from existing gold
data; the rest genuinely need manual labeling. Next step: decide
whether to scale up generation to more sentences, begin manual
labeling of the `needs_annotation` set in earnest, or continue
hardening if further trial batches surface more issues.
