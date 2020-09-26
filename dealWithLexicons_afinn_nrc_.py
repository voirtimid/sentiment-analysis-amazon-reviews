import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.stem import PorterStemmer


def get_positive_words():
    positive_words = []
    with open(r"lexicons/liu_positive-words.txt") as file:
        for line in file:
            data = line.split("\n")
            positive_words.append(data[0])

        return positive_words


def get_negative_words():
    negative_words_file = open("lexicons/liu_negative-words.txt", "r", encoding="ISO-8859-1")
    negative_words_context = negative_words_file.readlines()
    negative_words = []

    for line in negative_words_context:
        data = line.split("\n")
        negative_words.append(data[0])

    negative_words_file.close()
    return negative_words


def get_AFINN_lexicon():
    # word integer pairs
    #   20009 lines
    lexicon = dict()
    with open(r'lexicons/AFINN-111.txt') as f1:
        for line in f1:
            word, score = line.split('\t')
            lexicon[word] = float(score)
    return lexicon


def get_NRCVAD_lexicon():
    # Word	Valence	Arousal	Dominance
    #   2477 lines
    lexicon = dict()
    with open(r'lexicons/NRC-VAD.txt') as f1:
        next(f1)
        for line in f1:
            word, valence, arousal, dominance = line.split('\t')
            lexicon[word] = [float(valence), float(arousal), float(dominance)]
    return lexicon


def get_NRCAffectIntensity_lexicon():
    # word integer emotion  --> ['joy', 'fear', 'anger', 'sadness']
    #   5815 lines
    lexicon = dict()
    with open(r'lexicons/NRC-AffectIntensity.txt') as f1:
        for line in f1:
            word, score, emotion = line.split(' ')
            lexicon[word] = [float(score), emotion.strip()]
    return lexicon


afinn = get_AFINN_lexicon()
nrc_vad = get_NRCVAD_lexicon()
nrc_affect_intensity = get_NRCAffectIntensity_lexicon()
positive_words = get_positive_words()
negative_words = get_negative_words()


def wordslists_to_afinn_score_lists(wordslists):  # integer list of affinscores for words for each wordlist
    afinnscoreslist = []
    for wordlist in wordslists:
        afinnscr = []
        for word in wordlist:
            if word in afinn:
                afinnscr.append(afinn[word])
            else:
                afinnscr.append(0.0)  # default 0
        afinnscoreslist.append(afinnscr)
    return afinnscoreslist


def wordslists_to_summary_afinn_score(wordslists):  # integer for summary affinscore for each wordlist
    afinnscoreslist = []
    for wordlist in wordslists:
        afinnscr = 0.0
        for word in wordlist:
            if word in afinn:
                afinnscr = afinnscr + afinn[word]
        afinnscoreslist.append(afinnscr)
    return afinnscoreslist


def wordslists_to_nrcvad_score_lists(wordslists):  # [valence,arousal,dominance] vector for each word in each wordlist
    nrcvadscoreslist = []
    for wordlist in wordslists:
        nrcvadscr = []
        for word in wordlist:
            if word in nrc_vad:
                nrcvadscr.append(nrc_vad[word])
            else:
                nrcvadscr.append([0.0, 0.0, 0.0])  # default vector
        nrcvadscoreslist.append(nrcvadscr)
    return nrcvadscoreslist


def wordslists_to_nrcvad_summaryvectors_lists(wordslists):  # [valence,arousal,dominance] vector for each wordlist
    nrcvadscoreslist = []
    for wordlist in wordslists:
        valence = 0.0
        arousal = 0.0
        dominance = 0.0
        for word in wordlist:
            if word in nrc_vad:
                valence = valence + nrc_vad[word][0]
                arousal = arousal + nrc_vad[word][1]
                dominance = dominance + nrc_vad[word][2]
        cnt = len(wordlist)
        nrcvadscoreslist.append([valence / cnt, arousal / cnt, dominance / cnt])
    return nrcvadscoreslist


def wordslists_to_nrc_affin_score_lists(wordslists, intencodeemotion=False, onehotencodeemotion=False,
                                        flatten=True):  # [score,emotion] vector for each word in each wordlist, emotion can be string(FF),int(TF),onehot(FT)
    nrcafinnscoreslist = []
    emotion_integers = {'joy': 1, 'fear': 2, 'anger': 3, 'sadness': 4}
    emotion_onehots = {'joy': [1, 0, 0, 0], 'fear': [0, 1, 0, 0], 'anger': [0, 0, 1, 0], 'sadness': [0, 0, 0, 1]}

    for wordlist in wordslists:
        nrcafinnscr = []
        for word in wordlist:
            if word in nrc_affect_intensity:
                if intencodeemotion == False and onehotencodeemotion == False:
                    nrcafinnscr.append(nrc_affect_intensity[word])  # (score,'string emotion')
                elif intencodeemotion == True:
                    nrcafinnscr.append([nrc_affect_intensity[word][0], emotion_integers[
                        nrc_affect_intensity[word][1]]])  # (score,'int map for emotion')
                elif onehotencodeemotion == True and flatten == False:
                    nrcafinnscr.append([nrc_affect_intensity[word][0], emotion_onehots[
                        nrc_affect_intensity[word][1]]])  # (score,'onehot map for emotion')
                elif onehotencodeemotion == True and flatten == True:
                    res = [nrc_affect_intensity[word][0]]
                    res.extend(emotion_onehots[nrc_affect_intensity[word][1]])
                    nrcafinnscr.append(res)  # (score,'onehot map for emotion')
            else:
                if intencodeemotion == False and onehotencodeemotion == False:
                    nrcafinnscr.append([0.0, 'none'])  # (score,'string emotion')
                elif intencodeemotion == True:
                    nrcafinnscr.append([0.0, 0])  # (score,'int map for emotion')
                elif onehotencodeemotion == True and flatten == False:
                    nrcafinnscr.append([0.0, [0, 0, 0, 0]])  # (score,'onehot map for emotion')
                elif onehotencodeemotion == True and flatten == True:
                    nrcafinnscr.append([0.0, 0, 0, 0, 0])  # (score,'onehot map for emotion')
        nrcafinnscoreslist.append(nrcafinnscr)
    return nrcafinnscoreslist


def wordslists_to_nrc_affin_score_vectors(wordslists):  # [joy fear anger sadness] (onehot) * score
    nrcafinnscoreslist = []
    emotion_onehots = {'joy': [1, 0, 0, 0], 'fear': [0, 1, 0, 0], 'anger': [0, 0, 1, 0], 'sadness': [0, 0, 0, 1]}

    for wordlist in wordslists:
        nrcafinnscr = []
        for word in wordlist:
            if word in nrc_affect_intensity:
                value = nrc_affect_intensity[word][0]
                oh = emotion_onehots[nrc_affect_intensity[word][1]]
                res = [element * value for element in oh]
                nrcafinnscr.append(res)
            else:
                nrcafinnscr.append([0.0, 0.0, 0.0, 0.0])
        nrcafinnscoreslist.append(nrcafinnscr)
    return nrcafinnscoreslist


def wordslists_to_summary_nrc_affin_score(wordslists):  # [joy,fear,anger,sadness] averages vector per wordlist
    nrcafinnscoreslist = []
    for wordlist in wordslists:
        emotscores = {'joy': 0.0, 'fear': 0.0, 'anger': 0.0, 'sadness': 0.0}
        emotcnts = {'joy': 0, 'fear': 0, 'anger': 0, 'sadness': 0}
        for word in wordlist:
            if word in nrc_affect_intensity:
                emotscores[nrc_affect_intensity[word][1]] = emotscores[nrc_affect_intensity[word][1]] + \
                                                            nrc_affect_intensity[word][0]
                emotcnts[nrc_affect_intensity[word][1]] = emotcnts[nrc_affect_intensity[word][1]] + 1
        result = []
        for key in ['joy', 'fear', 'anger', 'sadness']:
            if emotcnts[key] == 0:
                result.append(0)
            else:
                result.append(emotscores[key] / emotcnts[key])
        nrcafinnscoreslist.append(result)
    return nrcafinnscoreslist


def get_summary_positive_negative_words(wordslist):
    result = []
    for wordlist in wordslist:
        positive = 0
        negative = 0

        for word in wordlist:
            if word in positive_words:
                positive += 1
            if word in negative_words:
                negative += 1
        result.append([positive / len(wordlist), negative / len(wordlist)])

    return result


def get_maximum_words(wordslist):
    return max(map(lambda item: len(item), wordslist))


def get_word_ratio(wordslist):
    max_length = get_maximum_words(wordslist)
    return list(map(lambda item: len(item) / max_length, wordslist))


def penn_to_wn(tag):
    """
    Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return wn.NOUN  # None


def get_sentiment(word, tag):
    """ returns list of pos neg and objective score. But returns empty list if not present in senti wordnet. """
    lemmatizer = WordNetLemmatizer()

    wn_tag = penn_to_wn(tag)
    if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
        return [0.0, 0.0, 0.0]

    lemma = lemmatizer.lemmatize(word, pos=wn_tag)
    if not lemma:
        return [0.0, 0.0, 0.0]

    synsets = wn.synsets(word, pos=wn_tag)
    if not synsets:
        return [0.0, 0.0, 0.0]  # before it was []

    # Take the first sense, the most common
    synset = synsets[0]
    swn_synset = swn.senti_synset(synset.name())

    return [swn_synset.pos_score(), swn_synset.neg_score(), swn_synset.obj_score()]


def wordlists_to_summary_posnegobjscore(wordlists):
    result = []
    for wordlist in wordlists:
        pos_val = nltk.pos_tag(wordlist)
        senti_val = [get_sentiment(x, y) for (x, y) in pos_val]
        dictionary = {0: 0.0, 1: 0.0, 2: 0.0}
        for sentval in senti_val:
            dictionary[0] = dictionary[0] + sentval[0]
            dictionary[1] = dictionary[1] + sentval[1]
            dictionary[2] = dictionary[2] + sentval[2]
        result.append([dictionary[0] / len(wordlist), dictionary[1] / len(wordlist), dictionary[2] / len(wordlist)])
    return result
