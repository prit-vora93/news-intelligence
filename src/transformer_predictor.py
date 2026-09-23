# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification


# MODEL_PATH = "models/newsmtc_distilbert_final"

# LABELS = {
#     0: "Negative",
#     1: "Neutral",
#     2: "Positive",
# }


# class TransformerPredictor:
#     def __init__(self, model_path=MODEL_PATH):
#         print("Loading tokenizer...")
#         self.tokenizer = AutoTokenizer.from_pretrained(model_path)

#         print("Loading model...")
#         self.model = AutoModelForSequenceClassification.from_pretrained(
#             model_path
#         )

#         # We are currently using CPU locally.
#         self.device = torch.device("cpu")
#         self.model.to(self.device)

#         self.model.eval()

#         print("Model loaded successfully.")

#     def mark_target(self, sentence, target):
#         """
#         Mark the target entity inside the sentence.
#         """

#         start = sentence.find(target)

#         if start == -1:
#             raise ValueError(
#                 f"Target '{target}' was not found in the sentence."
#             )

#         end = start + len(target)

#         marked_sentence = (
#             sentence[:start]
#             + " [TARGET] "
#             + sentence[start:end]
#             + " [/TARGET] "
#             + sentence[end:]
#         )

#         return marked_sentence.strip()

#     def predict(self, sentence, target):
#         """
#         Predict sentiment toward a specific target.
#         """

#         marked_sentence = self.mark_target(
#             sentence,
#             target
#         )

#         inputs = self.tokenizer(
#             marked_sentence,
#             return_tensors="pt",
#             truncation=True,
#             max_length=256
#         )

#         inputs = {
#             key: value.to(self.device)
#             for key, value in inputs.items()
#         }

#         with torch.no_grad():
#             outputs = self.model(**inputs)

#         probabilities = torch.softmax(
#             outputs.logits,
#             dim=-1
#         )[0]

#         predicted_class = torch.argmax(
#             probabilities
#         ).item()

#         confidence = probabilities[
#             predicted_class
#         ].item()

#         return {
#             "sentence": sentence,
#             "target": target,
#             "marked_sentence": marked_sentence,
#             "prediction": LABELS[predicted_class],
#             "confidence": confidence,
#             "probabilities": {
#                 LABELS[i]: probabilities[i].item()
#                 for i in range(3)
#             },
#         }


# if __name__ == "__main__":

#     predictor = TransformerPredictor()

#     # sentence = (
#     #     "President Trump praised the new economic policy."
#     # )

#     # target = "President Trump"

#     # result = predictor.predict(
#     #     sentence,
#     #     target
#     # )

#     # print("\nPrediction")
#     # print("----------")
#     # print("Sentence:", result["sentence"])
#     # print("Target:", result["target"])
#     # print("Marked:", result["marked_sentence"])
#     # print("Prediction:", result["prediction"])
#     # print(
#     #     "Confidence:",
#     #     f"{result['confidence'] * 100:.2f}%"
#     # )

#     # print("\nProbabilities")
#     # print("-------------")

#     # for label, probability in result["probabilities"].items():
#     #     print(
#     #         f"{label}: {probability * 100:.2f}%"
#     # )
#     examples = [
#         (
#             "President Trump praised the new economic policy.",
#             "President Trump"
#         ),
#         (
#             "Critics said President Trump handled the situation poorly.",
#             "President Trump"
#         ),
#         (
#             "The company announced strong profits today.",
#             "company"
#         ),
#     ]

#     for sentence, target in examples:

#         result = predictor.predict(
#             sentence,
#             target
#         )

#         print("\n" + "=" * 60)
#         print("Sentence:", sentence)
#         print("Target:", target)
#         print("Prediction:", result["prediction"])
#         print(
#             "Confidence:",
#             f"{result['confidence'] * 100:.2f}%"
#         )

#         print("Probabilities:")

#         for label, probability in result["probabilities"].items():
#             print(
#                 f"  {label}: "
#                 f"{probability * 100:.2f}%"
#             )

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_PATH = "models/newsmtc_distilbert_final"

LABELS = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}


class TransformerPredictor:

    def __init__(self, model_path=MODEL_PATH):

        print("Loading tokenizer...")
        # self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        print("Loading model...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path
        )

        self.device = torch.device("cpu")

        self.model.to(self.device)
        self.model.eval()

        print("Model loaded successfully.")

    def mark_target(
        self,
        sentence,
        target,
        target_from=None,
        target_to=None
    ):
        """
        Mark the exact target occurrence.

        If target_from and target_to are provided,
        they are preferred over string searching.
        """

        if target_from is not None and target_to is not None:

            start = int(target_from)
            end = int(target_to)

            actual_target = sentence[start:end]

            if actual_target != target:
                raise ValueError(
                    f"Target mismatch: offsets point to "
                    f"'{actual_target}', but target is '{target}'."
                )

        else:

            start = sentence.find(target)

            if start == -1:
                raise ValueError(
                    f"Target '{target}' was not found in sentence."
                )

            end = start + len(target)
        
        return (
            sentence[:start].rstrip()
            + " [TARGET] "
            + sentence[start:end]
            + " [/TARGET] "
            + sentence[end:].lstrip()
        ).strip()
    
    

    def predict(
        self,
        sentence,
        target,
        target_from=None,
        target_to=None
    ):
        """
        Predict sentiment toward a specific target.
        """

        marked_sentence = self.mark_target(
            sentence=sentence,
            target=target,
            target_from=target_from,
            target_to=target_to
        )

        inputs = self.tokenizer(
            marked_sentence,
            return_tensors="pt",
            truncation=True,
            max_length=256
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        # DistilBERT does not accept token_type_ids (an architectural
        # simplification vs. full BERT), but the configured
        # BertTokenizer includes it by default -- drop it before
        # calling the model.
        inputs.pop("token_type_ids", None)

        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

        predicted_class = torch.argmax(
            probabilities
        ).item()

        confidence = probabilities[
            predicted_class
        ].item()

        return {
            "sentence": sentence,
            "target": target,
            "marked_sentence": marked_sentence,
            "prediction": LABELS[predicted_class],
            "confidence": confidence,
            "probabilities": {
                LABELS[i]: float(probabilities[i])
                for i in range(3)
            }
        }


if __name__ == "__main__":

    predictor = TransformerPredictor()

    sentence = (
        "President Trump praised the new economic policy."
    )

    target = "President Trump"

    result = predictor.predict(
        sentence,
        target
    )

    print("\nPrediction")
    print("----------")
    print("Sentence:", result["sentence"])
    print("Target:", result["target"])
    print("Marked:", result["marked_sentence"])
    print("Prediction:", result["prediction"])
    print(
        "Confidence:",
        f"{result['confidence'] * 100:.2f}%"
    )

    print("\nProbabilities")
    print("-------------")

    for label, probability in result["probabilities"].items():

        print(
            f"{label}: "
            f"{probability * 100:.2f}%"
        )