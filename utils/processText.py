
def process_text(texts, context_length = 77):
    res = []
    for text in texts:
        text = text.split()
        if len(text) > 77:
            text = text[0:77]
            res.append(" ".join(text))
        else:
            res.append(" ".join(text))
    return res
    
if __name__ == "__main__":
    texts = ["a b c a", " a d d d d d d"]
    print(process_text(texts))