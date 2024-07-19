from transformers import AutoModel, AutoTokenizer


if __name__ == '__main__':
    models = ['bert-base-cased','bert-large-cased','roberta-base','roberta-large','bert-base-uncased','bert-large-uncased']
    for model_name in models:    
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(model_name,tokenizer.encode('positive negative yes neutral no true false'))