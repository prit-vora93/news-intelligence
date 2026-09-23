import spacy

nlp = spacy.load("en_core_web_trf")

sentences = [
    "Vice President JD Vance and other Trump administration officials said Tuesday they plan to remove 760,000 Affordable Care Act enrollees from public healthcare exchanges, alleging that the individuals were fraudulently enrolled in the program, or simply do not exist.",
    "Vance said the discovery and canceled subsidy payments for hundreds of thousands of people from the exchanges, part of an administration-wide crackdown on fraud, would result in $2.2 billion in cost savings.",
    "\"We're actually making sure that the people receiving Obamacare subsidies are actually entitled to receive them,\" said Vance, who leads a government task force to eliminate fraud.",
    "Vance said the cancellation of approximately 315,000 enrollments covering 760,000 people was done in part because the government was not checking whether people enrolled were eligible.",
    "Vance said another 419,000 enrollments will get additional verification to make sure they are eligible for the benefit.",
]

print("=" * 70)
print("ENTITY TAGS FOR 'Vance' IN EACH SENTENCE")
print("=" * 70)

for i, sentence in enumerate(sentences):
    doc = nlp(sentence)
    print(f"\n[Sentence {i}]: {sentence[:80]}...")
    for ent in doc.ents:
        if "vance" in ent.text.lower():
            print(f"  ENTITY: {ent.text!r}  label={ent.label_}")
    # Also check for any token containing "government" and its dependency info
    for token in doc:
        if "government" in token.text.lower():
            print(f"  TOKEN 'government': dep={token.dep_} head={token.head.text!r} ent_type={token.ent_type_!r}")
