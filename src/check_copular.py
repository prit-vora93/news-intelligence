import spacy

nlp = spacy.load("en_core_web_sm")

sentences = [
    "Trump is very emphatically not a seasoned diplomat.",
    "Roger was a great patriot.",
    "Handel was the top GOP vote-getter.",
]

for sentence in sentences:
    doc = nlp(sentence)
    print()
    print("=" * 70)
    print(sentence)
    print("=" * 70)
    for token in doc:
        print(
            f"{token.i:2} "
            f"{token.text:15} "
            f"POS={token.pos_:8} "
            f"TAG={token.tag_:8} "
            f"DEP={token.dep_:10} "
            f"HEAD={token.head.text:15} "
            f"HEAD_I={token.head.i}"
        )
