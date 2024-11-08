from transformers import AutoTokenizer


def list_tokens(model_name):
    # Load the tokenizer for the specified model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize the text
    tokens = tokenizer.tokenize("Unahitaji kuona kwangu, kaka. Nimejenga nyumba nzuri sana.")

    print(tokens)
    return


    # Retrieve the vocabulary dictionary
    vocab = tokenizer.get_vocab()
    print(len(vocab.items()))

    # Sort the tokens for better readability
    sorted_tokens = sorted(vocab.items(), key=lambda x: x[0])

    # # Print the tokens
    for token, index in vocab.items():
        print(f"Token: {token}, Index: {index}")



if __name__ == "__main__":
    # Specify the model name for Gemma2 7B
    model_name = "google/gemma-2-9b-it"  # Adjust this if the model name is different
    list_tokens(model_name)
