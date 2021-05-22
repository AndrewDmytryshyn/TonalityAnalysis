from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def get_sequences(tokenizer, x):
    sequences = tokenizer.texts_to_sequences(x)
    return pad_sequences(sequences, maxlen=26)


def tonality(str):
    model = Model()
    model.load_weights('cnn/cnn-trainable.hdf5')
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([str])
    x_test_seq1 = get_sequences(tokenizer, [str])
    predicted = model.predict(x_test_seq1)
    if predicted < 0.33:
        print("Negative, " + predicted)
        return 'neg'
    elif predicted < 0.66:
        print("Neutral, " + predicted)
        return 'neu'
    else:
        print("Positive, " + predicted)
        return 'pos'
