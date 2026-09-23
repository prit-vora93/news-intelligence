import spacy

nlp = spacy.load("en_core_web_sm")

sentence = "Amazon, Google, and Microsoft are all investing in AI infrastructure."
doc = nlp(sentence)

print()
print("DEPENDENCY TREE")
print("=" * 70)
for token in doc:
    print(
        f"{token.i:2} "
        f"{token.text:15} "
        f"POS={token.pos_:8} "
        f"DEP={token.dep_:10} "
        f"HEAD={token.head.text:15} "
        f"HEAD_I={token.head.i}"
    )

print()
print("ENTITIES")
print("=" * 70)
for ent in doc.ents:
    print(f"  {ent.text!r:25} label={ent.label_}")
