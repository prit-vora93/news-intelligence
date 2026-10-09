path = "/home/prit/news_intelligence/src/transformer_predictor.py"

with open(path) as f:
    content = f.read()

old = '''        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():'''

new = '''        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        # DistilBERT does not accept token_type_ids (an architectural
        # simplification vs. full BERT), but the configured
        # BertTokenizer includes it by default -- drop it before
        # calling the model.
        inputs.pop("token_type_ids", None)

        with torch.no_grad():'''

if old not in content:
    print("ERROR: expected code block not found -- did not modify file")
    print("Paste back the current content of predict() so I can adjust.")
else:
    content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)
    print("Patched successfully")
