import torch
import torch.nn as nn
import pickle
import nltk
import random
import os


try:
    nltk.data.find('tokenizers/punkt_tab')
    from nltk.tokenize import word_tokenize
except LookupError:
    try:
        nltk.data.find('tokenizers/punkt')
        from nltk.tokenize import word_tokenize
    except LookupError:
        # Резерв: простой split + очистка
        def word_tokenize(text):
            return text.lower().strip().replace('.', ' .').replace(',', ' ,').replace('?', ' ?').replace('!', ' !').split()


def create_encoder(vocab_size, embed_size, hidden_size):
    embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
    lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
    return embedding, lstm

def create_decoder(vocab_size, embed_size, hidden_size):
    embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
    lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
    fc = nn.Linear(hidden_size, vocab_size)
    return embedding, lstm, fc


def generate_response(
    user_input,
    enc_emb, enc_lstm,
    dec_emb, dec_lstm, dec_fc,
    word_to_idx, idx_to_word,
    max_len=20,
    device=torch.device("cpu")
):

    tokens = word_tokenize(user_input.lower().strip())
    indices = [word_to_idx.get(w, word_to_idx["<UNK>"]) for w in tokens]
    indices = indices[:max_len - 1]
    indices.append(word_to_idx["<EOS>"])
    while len(indices) < max_len:
        indices.append(word_to_idx["<PAD>"])
    
    src = torch.tensor([indices], dtype=torch.long).to(device)  # [1, T]


    with torch.no_grad():
        enc_out = enc_emb(src)
        _, (hidden, cell) = enc_lstm(enc_out)


        dec_input = torch.tensor([[word_to_idx["<SOS>"]]], device=device)
        output_tokens = []

        for _ in range(max_len - 1):
            dec_emb_out = dec_emb(dec_input)
            dec_out, (hidden, cell) = dec_lstm(dec_emb_out, (hidden, cell))
            pred = dec_fc(dec_out.squeeze(1))
            top1 = pred.argmax(1)
            word_id = top1.item()
            
            if word_id == word_to_idx["<EOS>"] or word_id == word_to_idx["<PAD>"]:
                break
            output_tokens.append(idx_to_word.get(word_id, "<UNK>"))
            dec_input = top1.unsqueeze(0)

    return " ".join(output_tokens)


def load_model(model_path="chatbot_model.pth", device=torch.device("cpu")):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель не найдена: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    # Восстановление параметров
    vocab_size = checkpoint["vocab_size"]
    embed_size = checkpoint["embed_size"]
    hidden_size = checkpoint["hidden_size"]
    word_to_idx = checkpoint["word_to_idx"]
    idx_to_word = checkpoint["idx_to_word"]

    # Создание модели
    enc_emb, enc_lstm = create_encoder(vocab_size, embed_size, hidden_size)
    dec_emb, dec_lstm, dec_fc = create_decoder(vocab_size, embed_size, hidden_size)

    # Загрузка весов
    enc_emb.load_state_dict(checkpoint["encoder_embedding"])
    enc_lstm.load_state_dict(checkpoint["encoder_lstm"])
    dec_emb.load_state_dict(checkpoint["decoder_embedding"])
    dec_lstm.load_state_dict(checkpoint["decoder_lstm"])
    dec_fc.load_state_dict(checkpoint["decoder_fc"])

    # Eval mode
    enc_emb.eval()
    enc_lstm.eval()
    dec_emb.eval()
    dec_lstm.eval()
    dec_fc.eval()

    return (enc_emb, enc_lstm, dec_emb, dec_lstm, dec_fc, word_to_idx, idx_to_word)


def main():
    device = torch.device("cpu")
    print("Загрузка модели...")
    try:
        model = load_model("chatbot_model.pth", device)
        enc_emb, enc_lstm, dec_emb, dec_lstm, dec_fc, word_to_idx, idx_to_word = model
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return

    print("\nЧат-бот из 'Игры престолов'. Введите 'выход', чтобы завершить.\n")
    
    while True:
        try:
            user_input = input("Ты: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("выход", "exit", "quit"):
                print("Winter is coming...")
                break

            response = generate_response(
                user_input,
                enc_emb, enc_lstm,
                dec_emb, dec_lstm, dec_fc,
                word_to_idx, idx_to_word,
                max_len=20,
                device=device
            )

            if not response.strip():
                response = random.choice([
                    "Hmm...", "I see.", "Go on.", "And then?", "Tell me more."
                ])

            print(f"Бот: {response}")

        except KeyboardInterrupt:
            print("\nДо свидания!")
            break
        except Exception as e:
            print(f"[Ошибка] {e}")

if __name__ == "__main__":
    main()
