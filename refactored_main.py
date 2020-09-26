from dealWithLexicons_afinn_nrc_ import *
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix

train_data_filename = "datasets/test.ft.txt"
labels_train_data = []
sentences_train_data = []

with open(r"datasets/test.ft.txt") as train_data_file:
    for line in train_data_file:
        data = line.split(" ", 1)
        labels_train_data.append(1 if data[0].__eq__("__label__2") else 0)
        sentences_train_data.append(data[1])

# Take first 100 sentences and labels
# labels_train_data = labels_train_data[:10000]
# sentences_train_data = sentences_train_data[:10000]

tokenized_sentences = []
for i in tqdm(range(0, len(sentences_train_data))):
    sentence = sentences_train_data[i].lower()
    tokenized_sentences.append(word_tokenize(sentence))

print("Started tokenizing")
afinnscoreslist = wordslists_to_summary_afinn_score(tokenized_sentences)
nrcvadscoreslist = wordslists_to_nrcvad_summaryvectors_lists(tokenized_sentences)
nrcafinnscoreslist = wordslists_to_summary_nrc_affin_score(tokenized_sentences)
positive_negative = get_summary_positive_negative_words(tokenized_sentences)
word_ratio = get_word_ratio(tokenized_sentences)
posnegobjscore = wordlists_to_summary_posnegobjscore(tokenized_sentences)
print("Finished tokenizing")

final_input = []
for i in range(400000):
    temp = [afinnscoreslist[i], word_ratio[i]]
    temp.extend(nrcvadscoreslist[i])
    temp.extend(nrcafinnscoreslist[i])
    temp.extend(positive_negative[i])
    temp.extend(posnegobjscore[i])
    final_input.append(temp)

np_sentence_scores = np.array(final_input)
np_labels = np.array(labels_train_data)

X_train, X_test, y_train, y_test = train_test_split(np_sentence_scores, np_labels, test_size=0.33, random_state=42)

model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(14,)))
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

preds = model.predict(X_test)

print('Accuracy score: {:0.4}'.format(accuracy_score(y_test, 1 * (preds > 0.5))))
print('F1 score: {:0.4}'.format(f1_score(y_test, 1 * (preds > 0.5))))
print('ROC AUC score: {:0.4}'.format(roc_auc_score(y_test, preds)))

preds_normalized = []

for el in preds:
    if el > 0.5:
        preds_normalized.append(1)
    else:
        preds_normalized.append(0)
print('Confusion matrix:' + confusion_matrix(y_test, preds_normalized))
