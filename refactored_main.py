from dealWithLexicons_afinn_nrc_ import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, roc_curve, auc

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


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
            TP += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
            FP += 1
        if y_actual[i] == y_hat[i] == 0:
            TN += 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
            FN += 1

    return TP, FP, TN, FN


preds_normalized = []

for el in preds:
    if el > 0.5:
        preds_normalized.append(1)
    else:
        preds_normalized.append(0)
print('Confusion matrix:' + str(perf_measure(y_test, preds_normalized)))

fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
plt.savefig("plots/roc_curve.png")
