from g2pk import G2p

from jamo import h2j, j2hcj

from text_normalize import text_normalize

JAMO_LIST = [
    'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 
    'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
    'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ',
    'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ'
]
SYM_LIST = [
    '.', '!', '?', '~', ','
]
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'

vocab = [PAD_TOKEN, UNK_TOKEN] + SYM_LIST + JAMO_LIST 
jamo2id = {j: idx for idx, j in enumerate(vocab)}
id2jamo = {idx: j for j, idx in jamo2id.items()}

def text_embedding(input):
    text_ids = []
    g2p = G2p()
    normalized = text_normalize(input)
    pron = g2p(normalized)
    
    jamo_sentence = h2j(pron)
    jamos = list(jamo_sentence)
    print(f'Pron: {jamos}')
    
    for jm in jamos:
        jm = j2hcj(jm)
        if jm in jamo2id: text_ids.append(jamo2id[jm])
        else: text_ids.append(jamo2id[UNK_TOKEN])
    
    return text_ids
    

if __name__ == '__main__':
    text = input()
    print(f'Symbol: {text_embedding(text)}')