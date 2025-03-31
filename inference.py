import torch
import torch.utils
import os
from transformers import AutoTokenizer
from model import Transformer


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
    tokenizer.add_special_tokens({"bos_token": "<s>"})

    src_vocab_size = tokenizer.vocab_size + len(tokenizer.special_tokens_map)
    dst_vocab_size = tokenizer.vocab_size + len(tokenizer.special_tokens_map)
    pad_idx = tokenizer.pad_token_id
    print(pad_idx)
    
    # Model hyperparameters
    d_model = 512
    num_layers = 6
    heads = 8
    d_ff = 1024
    dropout = 0.1
    max_seq_len = 40
    batch_size = 1

    model = Transformer(
        src_vocab_size,
        dst_vocab_size,
        pad_idx,
        d_model,
        num_layers,
        heads,
        d_ff,
        dropout,
        max_seq_len
    )
    model.to(device)

    if os.path.exists("./model.pt"):
        model.load_state_dict(torch.load("./model.pt"))
    
    # input_ = "I am Tom."
    # input_ = "What do you like doing in your spare time?"
    # input_ = [
        # "What do you like doing in your spare time?",
        # "I have two nieces.",
        # "My brother won't be at home tomorrow.",
        # "He translated the verse into English.",
        # "I need you.",
        # "I am Tom.",
    # ]
    while True:
        input_ = input("Enter text to translate (or 'q' to quit): ")
        if input_.lower() == 'q':
            break
            
        input_in = tokenizer(
            input_,
            padding="max_length", 
            max_length=max_seq_len,
            truncation=True,
            return_tensors="pt"
        )["input_ids"]
        input_in = input_in.to(device)
        print("Input size:", input_in.size())

        de_in = torch.ones(
            batch_size, max_seq_len, dtype=torch.long
        ).to(device) * pad_idx
        de_in[:, 0] = tokenizer.bos_token_id
        print("Output size:", de_in.size())

        model.eval()
        with torch.no_grad():
            for i in range(1, de_in.shape[1]):
                pred_ = model(input_in, de_in)
                for j in range(batch_size):
                    de_in[j, i] = torch.argmax(pred_[j, i-1], dim=-1)

        for de in de_in:
            out = []
            for i in de:
                if i == tokenizer.eos_token_id:
                    break
                out.append(tokenizer.decode(i))
            print("Translation:", " ".join(out))
