from nltk.tokenize import word_tokenize
from keras.models import Sequential
from keras.layers import Dense, Dropout

train_data_filename = "datasets/train.ft.txt"

train_data_file = open(train_data_filename, "r")
train_data_lines = train_data_file.readlines()

labels_train_data = []
sentences_train_data = []

for line in train_data_lines:
    data = line.split(" ", 1)
    labels_train_data.append("good" if data[0].__eq__("__label__2") else "bad")
    sentences_train_data.append(data[1])

train_data_file.close()
# Take first 100 sentences and labels
labels_train_data = labels_train_data[:100]
sentences_train_data = sentences_train_data[:100]

tokenized_sentences = []
for sentence in sentences_train_data:
    sentence = sentence.lower()
    tokenized_sentences.append(word_tokenize(sentence))

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

sentence_scores = []
for sentence in tokenized_sentences:
    score = 0
    positive = 0
    negative = 0
    for word in sentence:
        if word in afinn_dictionary.keys():
            score += afinn_dictionary[word]
        if word in positive_words:
            positive += 1
        if word in negative_words:
            negative += 1
    sentence_scores.append([score, positive, negative])

print(sentence_scores[0])

model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(100, 3)))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(sentence_scores, labels_train_data, batch_size=32, epochs=15, verbose=0)

