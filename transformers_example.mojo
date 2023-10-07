from python.python import Python


fn main():
    try:
        let transformers = Python.import_module("transformers")
        let tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
        let model = transformers.AutoModelForMaskedLM.from_pretrained("bert-base-cased")

        let unmasker = transformers.pipeline(
            "fill-mask", "bert-base-multilingual-cased"
        )

        print(unmasker("Hello I'm a [MASK] model."))

    except e:
        print(e)
