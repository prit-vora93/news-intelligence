import spacy

nlp = spacy.load("en_core_web_sm")

sentences = [
    # Suspected simple PP-embedding case
    "He showed he's a man of action by flying to Mexico on a moment's notice.",

    # Suspected fronted-PP case
    "At 21st Century Fox we will always be enormously grateful for the great business he built.",

    # Uncertain: might be PP-embedding OR a coordination-parsing issue we
    # already have logic for -- need to check before assuming.
    "Hillary Clinton blamed the Democratic National Committee, Facebook, and conspiracy site Infowars for her election defeat.",
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
            f"{token.text:20} "
            f"POS={token.pos_:8} "
            f"DEP={token.dep_:10} "
            f"HEAD={token.head.text:15} "
            f"HEAD_I={token.head.i}"
        )
