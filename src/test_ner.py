import spacy


nlp = spacy.load("en_core_web_sm")


examples = [
    "Winner wrote that she had a 30-minute private meeting with the Republican lawmaker's state policy director.",
    "She also recently referred to President Trump as a piece of shit because of his position on the Dakota Access Pipeline protests.",
    "Hillary Clinton blamed the Democratic National Committee, Facebook, and conspiracy site Infowars for her election defeat.",
]


for text in examples:

    doc = nlp(text)

    print("\n" + "=" * 70)
    print("TEXT")
    print(text)

    print("\nENTITIES")
    for ent in doc.ents:
        print(
            f"{ent.text:30} "
            f"{ent.label_:10} "
            f"({ent.start_char}, {ent.end_char})"
        )