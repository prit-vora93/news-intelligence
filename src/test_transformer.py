# from src.transformer_predictor import TransformerPredictor
from transformer_predictor import TransformerPredictor


def main():
    predictor = TransformerPredictor()

    examples = [
        (
            "President Trump praised the new economic policy.",
            "President Trump",
        ),
        (
            "Critics said President Trump handled the situation poorly.",
            "President Trump",
        ),
        (
            "The company announced strong profits today.",
            "company",
        ),
    ]

    print("\nTransformer Prediction Tests")
    print("=" * 60)

    for sentence, target in examples:

        result = predictor.predict(
            sentence=sentence,
            target=target,
        )

        print("\n" + "-" * 60)
        print("Sentence:", sentence)
        print("Target:", target)
        print("Marked:", result["marked_sentence"])
        print("Prediction:", result["prediction"])
        print(
            "Confidence:",
            f"{result['confidence'] * 100:.2f}%"
        )

        print("Probabilities:")

        for label, probability in result["probabilities"].items():
            print(
                f"  {label}: "
                f"{probability * 100:.2f}%"
            )


if __name__ == "__main__":
    main()