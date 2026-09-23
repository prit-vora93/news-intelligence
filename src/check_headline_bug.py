import spacy

nlp = spacy.load("en_core_web_sm")

sentence = "Trump Picks Jeff Sessions, Senator Accused Of Racism, For Attorney General"

doc = nlp(sentence)

print("=" * 70)
print("ENTITIES RECOGNIZED")
print("=" * 70)
for ent in doc.ents:
    print(f"  {ent.text!r:50} label={ent.label_}")

print()
print("=" * 70)
print("PER-TOKEN POS/DEP")
print("=" * 70)
for token in doc:
    print(
        f"  {token.text!r:15} POS={token.pos_:8} "
        f"TAG={token.tag_:6} DEP={token.dep_:10} "
        f"ent_type_={token.ent_type_!r}"
    )
