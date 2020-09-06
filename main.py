import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

train_data_filename = "datasets/test.ft.txt"

train_data_file = open(train_data_filename, "r")
train_data_lines = train_data_file.readlines()

labels_train_data = []
sentences_train_data = []

for line in train_data_lines:
    data = line.split(" ", 1)
    labels_train_data.append(1 if data[0].__eq__("__label__2") else 0)
    sentences_train_data.append(data[1])

train_data_file.close()

# Take first 100 sentences and labels
labels_train_data = labels_train_data[:10000]
sentences_train_data = sentences_train_data[:10000]

tokenized_sentences = []
for i in tqdm(range(0, len(sentences_train_data))):
    sentence = sentences_train_data[i].lower()
    tokenized_sentences.append(word_tokenize(sentence))

# average_length = math.floor(sum(map(lambda x: len(x), tokenized_sentences)) / len(tokenized_sentences))
# print(average_length)
#
# temp_tokenized_sentences = []
# for i in tqdm(range(len(tokenized_sentences))):
#     sentence = tokenized_sentences[i]
#     if len(sentence) < average_length:
#         sentence.extend(" " * (average_length - len(sentence)))
#     else:
#         sentence = sentence[:average_length]
#     temp_tokenized_sentences.append(sentence)
#
# tokenized_sentences = temp_tokenized_sentences

afinn_file = open("lexicons/AFINN-111.txt", "r")
afinn_context = afinn_file.readlines()
afinn_dictionary = {}
for line in afinn_context:
    data = line.split("\t", 1)
    afinn_dictionary[data[0]] = int(data[1])

afinn_file.close()

positive_words_file = open("lexicons/liu_positive-words.txt", "r")
positive_words_context = positive_words_file.readlines()
positive_words = []

for line in positive_words_context:
    data = line.split("\n")
    positive_words.append(data[0])

positive_words_file.close()

negative_words_file = open("lexicons/liu_negative-words.txt", "r", encoding="ISO-8859-1")
negative_words_context = negative_words_file.readlines()
negative_words = []

for line in negative_words_context:
    data = line.split("\n")
    negative_words.append(data[0])

negative_words_file.close()

nrc_vad_file = open("lexicons/NRC-VAD.txt", "r")
nrc_vad_context = nrc_vad_file.readlines()
nrc_vad_dictionary = {}
skip_first = True
for line in nrc_vad_context:
    if not skip_first:
        data = line.split("\t", 1)
        nrc_data = [float(item) for item in data[1].split("\t")]
        nrc_vad_dictionary[data[0]] = nrc_data
    else:
        skip_first = False

nrc_vad_file.close()

nrc_aff_file = open("lexicons/NRC-AffectIntensity.txt", "r")
nrc_aff_context = nrc_aff_file.readlines()
nrc_aff_dictionary = {}
for line in nrc_aff_context:
    data = line.split(" ")
    data[2] = data[2].split("\n")[0]
    if data[0] in nrc_aff_dictionary.keys():
        temp = nrc_aff_dictionary[data[0]]
        temp.append((data[1], data[2]))
        nrc_aff_dictionary[data[0]] = temp
    else:
        nrc_aff_dictionary[data[0]] = [(data[1], data[2])]

nrc_aff_file.close()

sentence_scores = []
for i in tqdm(range(len(tokenized_sentences))):
    sentence = tokenized_sentences[i]
    score = 0
    positive = 0
    negative = 0
    nrc_vad_dict = {"Valence": 0, "Arousal": 0, "Dominance": 0}
    nrc_counter = 0
    nrc_aff_dict = {"anger": 0, "fear": 0, "sadness": 0, "joy": 0}
    nrc_aff_counter = 0
    afinn_counter = 0
    sentence_scores_temp = []
    for word in sentence:
        if word in afinn_dictionary.keys():
            score += afinn_dictionary[word]
            afinn_counter += 1
        if word in positive_words:
            positive += 1
        if word in negative_words:
            negative += 1
        if word in nrc_vad_dictionary.keys():
            nrc_list = nrc_vad_dictionary[word]
            nrc_vad_dict["Valence"] += nrc_list[0]
            nrc_vad_dict["Arousal"] += nrc_list[1]
            nrc_vad_dict["Dominance"] += nrc_list[2]
            nrc_counter += 1
        if word in nrc_aff_dictionary:
            data = nrc_aff_dictionary[word]
            for el in data:
                if el[1] == "anger":
                    nrc_aff_dict["anger"] += float(el[0])
                if el[1] == "fear":
                    nrc_aff_dict["fear"] += float(el[0])
                if el[1] == "sadness":
                    nrc_aff_dict["sadness"] += float(el[0])
                if el[1] == "joy":
                    nrc_aff_dict["joy"] += float(el[0])
                nrc_aff_counter += 1
    sentence_scores_temp.extend([positive, negative])
    if nrc_aff_counter == 0:
        sentence_scores_temp.extend([0, 0, 0, 0])
    else:
        sentence_scores_temp.extend([nrc_aff_dict["anger"] / nrc_aff_counter,
                                     nrc_aff_dict["fear"] / nrc_aff_counter,
                                     nrc_aff_dict["sadness"] / nrc_aff_counter,
                                     nrc_aff_dict["joy"] / nrc_aff_counter,
                                     ])
    if nrc_counter == 0:
        sentence_scores_temp.extend([0, 0, 0])
    else:
        sentence_scores_temp.extend([nrc_vad_dict["Valence"] / nrc_counter,
                                     nrc_vad_dict["Arousal"] / nrc_counter,
                                     nrc_vad_dict["Dominance"] / nrc_counter])
    if afinn_counter == 0:
        sentence_scores_temp.extend([0])
    else:
        sentence_scores_temp.extend([score / afinn_counter])
    sentence_scores.append(sentence_scores_temp)

print(sentence_scores[0])
np_sentence_scores = np.array(sentence_scores)
np_labels = np.array(labels_train_data)

X_train, X_test, y_train, y_test = train_test_split(np_sentence_scores, np_labels, test_size=0.33, random_state=42)

model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(10,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop')
model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test))

preds = model.predict_classes(X_test)

print('Accuracy score: {:0.4}'.format(accuracy_score(y_test, 1 * (preds > 0.5))))
print('F1 score: {:0.4}'.format(f1_score(y_test, 1 * (preds > 0.5))))
print('ROC AUC score: {:0.4}'.format(roc_auc_score(y_test, preds)))
