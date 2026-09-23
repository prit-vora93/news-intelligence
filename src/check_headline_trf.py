import spacy

try:
    nlp = spacy.load("en_core_web_trf")
except OSError:
    print("en_core_web_trf is not installed.")
    print("Install it with:")
    print("  pip install spacy-transformers --break-system-packages")
    print("  python -m spacy download en_core_web_trf")
    print("Then re-run this script.")
    raise SystemExit(1)

sentence = "Trump Picks Jeff Sessions, Senator Accused Of Racism, For Attorney General"

doc = nlp(sentence)

print("=" * 70)
print("ENTITIES RECOGNIZED (en_core_web_trf)")
print("=" * 70)
for ent in doc.ents:
    print(f"  {ent.text!r:50} label={ent.label_}")

print()
print("=" * 70)
print("PER-TOKEN POS/DEP (en_core_web_trf)")
print("=" * 70)
for token in doc:
    print(
        f"  {token.text!r:15} POS={token.pos_:8} "
        f"TAG={token.tag_:6} DEP={token.dep_:10} "
        f"ent_type_={token.ent_type_!r}"
    )
