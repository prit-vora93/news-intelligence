import json

records = [json.loads(line) for line in open("dataset_final.jsonl", encoding="utf-8")]

check_sentences = [
    "Marco Rubio, R-Fla., is publicly criticizing the same tax reform bill that he voted for.",
    "John McCain, R-Ariz., blasted President Donald Trump's decision to pardon a former Arizona sheriff who was accused of racially profiling Latinos.",
    "Former Florida governor Jeb Bush (R) said Obama won reelection by \"dividing the country.\"",
]

for target_sentence in check_sentences:
    print("=" * 70)
    print(f"SENTENCE: {target_sentence[:80]}...")
    print("=" * 70)
    matches = [r for r in records if r["sentence"] == target_sentence]
    for r in matches:
        print(f"  target={r['target_text']!r}  role={r['role']}  embedded_entities={r['embedded_entities']}")
    print()
