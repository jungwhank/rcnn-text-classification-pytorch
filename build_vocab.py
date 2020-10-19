from collections import Counter


def build_dictionary(texts, vocab_size):
    counter = Counter()
    SPECIAL_TOKENS = ['<PAD>', '<UNK>']

    for word in texts:
        counter.update(word)

    words = [word for word, count in counter.most_common(vocab_size - len(SPECIAL_TOKENS))]
    words = SPECIAL_TOKENS + words
    word2idx = {word: idx for idx, word in enumerate(words)}

    return word2idx