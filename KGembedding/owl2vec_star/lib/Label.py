import re
from nltk import word_tokenize


def URI_parse(uri):
    """Parse a URI: remove the prefix, parse the name part (Camel cases are plit)"""
    if '#' not in uri:
        ind = uri[::-1].index('/')
        name = uri[-ind:]
    else:
        name = re.sub("http[a-zA-Z0-9:/._-]+#", "", uri)

    name = name.replace('_', ' ').replace('-', ' ').replace('.', ' ').replace('/', ' '). \
        replace('"', ' ').replace("'", ' ')
    words = []
    for item in name.split():
        matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', item)
        for m in matches:
            word = m.group(0)
            words.append(word.lower())
#            if word.isalpha():
#                words.append(word.lower())
    return words


def pre_process_words(words):
    text = ' '.join([re.sub(r'https?:\/\/.*[\r\n]*', '', word, flags=re.MULTILINE) for word in words])
    tokens = word_tokenize(text)
    # processed_tokens = [token.lower() for token in tokens if token.isalpha()]
    processed_tokens = [token.lower() for token in tokens]
    return processed_tokens
