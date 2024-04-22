# creat a funciton to extract essential words
def extract_essential_words(df):
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    from nltk.probability import FreqDist
    import string
    import re

    ''' 
    The function extracts all essential words from the entire comments column,
    pick up the most frequent words to form the vocabulary for word vectorization

    '''
    # preloaded
    essential_words = []
    stem_fit = WordNetLemmatizer()
    # concat the entire column for the whole vocabulary
    for i, comment in enumerate(df.comments):
        try:
            # tokenise the sentence
            comment = comment.lower().translate(str.maketrans('', '', string.punctuation))
            words = word_tokenize(comment)
            # stem the words
            stem_words = [stem_fit.lemmatize(wd) for wd in words if wd not in stopwords.words('english')]
            if len(stem_words) != 0:
                df.comment[i] = stem_words
                [essential_words.append(wd) for wd in stem_words]
            else:
                df.comment[i] = np.nan
        except:
            df.comments[i] = np.nan
    # get the most frequenct words
    print(essential_words)
    freq_dist = FreqDist(essential_words)
    essential_words = freq_dist.most_common(10)
    print(essential_words)
    essential_words, _ = list(zip(*essential_words))
    essential_words = list(essential_words)
    # start to extract
    for i, comment in enumerate(df.comments):
        if comment is not np.nan:
            df.comments[i] = [wd for wd in comment if wd in essential_words]
        else:
            pass

    return essential_words