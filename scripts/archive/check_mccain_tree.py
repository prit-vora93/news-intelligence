import spacy

nlp = spacy.load("en_core_web_trf")

sentence = "John McCain, R-Ariz., blasted President Donald Trump's decision to pardon a former Arizona sheriff who was accused of racially profiling Latinos."

doc = nlp(sentence)

print("=" * 70)
print("PER-TOKEN DEP STRUCTURE (first 15 tokens)")
print("=" * 70)
for token in doc[:15]:
    print(
        f"{token.i:2} {token.text:12} POS={token.pos_:8} "
        f"DEP={token.dep_:10} HEAD={token.head.text:12} "
        f"ent_type_={token.ent_type_!r}"
    )
