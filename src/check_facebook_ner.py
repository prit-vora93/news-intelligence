import spacy

nlp = spacy.load("en_core_web_sm")

sentence = (
    "Hillary Clinton blamed the Democratic National Committee, "
    "Facebook, and conspiracy site Infowars for her election defeat."
)

doc = nlp(sentence)

print("=" * 70)
print("ENTITIES RECOGNIZED")
print("=" * 70)
for ent in doc.ents:
    print(f"  {ent.text!r:40} label={ent.label_}")

print()
print("=" * 70)
print("PER-TOKEN ent_type_ (to check specific words)")
print("=" * 70)
for token in doc:
    marker = " <-- check this" if token.text in (
        "Facebook", "Committee", "Infowars", "site", "conspiracy"
    ) else ""
    print(f"  {token.text!r:20} ent_type_={token.ent_type_!r:10} dep_={token.dep_}{marker}")
