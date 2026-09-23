import json

records = [json.loads(line) for line in open("dataset_v1.jsonl", encoding="utf-8")]

print("=== Check 1: any bare 'who'/'which'/'whom'/'whose' targets left? ===")
relative_pronoun_leaks = [
    r for r in records
    if r["target_text"].strip().lower() in ("who", "which", "whom", "whose")
]
if relative_pronoun_leaks:
    print(f"STILL PRESENT ({len(relative_pronoun_leaks)}):")
    for r in relative_pronoun_leaks:
        print(f"  {r['target_text']!r} -- {r['sentence'][:80]}")
else:
    print("CLEAN -- no relative pronouns leaking through as targets")

print()
print("=== Check 2: any dangling ' 's' spacing artifacts left? ===")
spacing_artifacts = [
    r for r in records
    if " 's" in r["target_text"] or " \u2019s" in r["target_text"]
]
if spacing_artifacts:
    print(f"STILL PRESENT ({len(spacing_artifacts)}):")
    for r in spacing_artifacts:
        print(f"  {r['target_text']!r} -- {r['sentence'][:80]}")
else:
    print("CLEAN -- no dangling possessive spacing artifacts")

print()
print("=== Check 3: find the specific Hillary sentence again to confirm ===")
for r in records:
    if "Hillary" in r["target_text"]:
        print(f"  {r['target_text']!r} (role={r['role']}, label_source={r['label_source']})")
