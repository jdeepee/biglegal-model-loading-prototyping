from flask import Flask, request, session
import numpy as np
import tensorflow as tf
import pickle as pkl
from model import Model
from prepare_file import prepare_file
from glove_model import GloveVec
import uuid, os, sys

app = Flask(__name__)

def find_max_length(file_name):
    temp_len = 0
    max_length = 0
    for line in open(file_name):
        if line in ['\n', '\r\n']:
            if temp_len > max_length:
                max_length = temp_len
            temp_len = 0
        else:
            temp_len += 1
    return max_length

def pos(tag):
    one_hot = np.zeros(5)
    if tag == 'NN' or tag == 'NNS':
        one_hot[0] = 1
    elif tag == 'FW':
        one_hot[1] = 1
    elif tag == 'NNP' or tag == 'NNPS':
        one_hot[2] = 1
    elif 'VB' in tag:
        one_hot[3] = 1
    else:
        one_hot[4] = 1

    return one_hot

def chunk(tag):
    one_hot = np.zeros(5)

    if 'NP' in tag:
        one_hot[0] = 1
    elif 'VP' in tag:
        one_hot[1] = 1
    elif 'PP' in tag:
        one_hot[2] = 1
    elif tag == 'O':
        one_hot[3] = 1
    else:
        one_hot[4] = 1

    return one_hot

def capital(word):
    if ord('A') <= ord(word[0]) <= ord('Z'):
        return np.array([1])
    else:
        return np.array([0])

def get_input(model, word_dim, input_file, sentence_length=350):
    print('processing %s' % input_file)
    word = []
    tag = []
    sentence = []
    sentence_tag = []

    if sentence_length == -1:
        max_sentence_length = find_max_length(input_file)
    else:
        max_sentence_length = sentence_length

    sentence_length = 0

    print("max sentence length is %d" % max_sentence_length)

    for line in open(input_file):
        if line in ['\n', '\r\n']:
            for _ in range(max_sentence_length - sentence_length):
                tag.append(np.array([0] * 10))
                temp = np.array([0 for _ in range(word_dim + 11)])
                word.append(temp)
            sentence.append(word)
            sentence_tag.append(np.array(tag))
            sentence_length = 0
            word = []
            tag = []

        else:
            assert (len(line.split()) == 4)
            sentence_length += 1
            temp = model[line.split()[0]]
            assert len(temp) == word_dim
            temp = np.append(temp, pos(line.split()[1]))  # adding pos embeddings
            temp = np.append(temp, chunk(line.split()[2]))  # adding chunk embeddings
            temp = np.append(temp, capital(line.split()[0]))  # adding capital embedding
            word.append(temp)

    assert (len(sentence) == len(sentence_tag))
    return sentence 

def convert_file(file):
    word_dim = 311
    input_file = file

    output_array = get_input(model_glove, word_dim, input_file)
    return output_array

def to_list(prediction, length):
    list_location = [[], [], [], [], [], [], [], [], []]
    current_line = 0
    prediction = np.argmax(prediction, 2)
    print prediction.shape
    print length.shape 

    for i in range(len(prediction)):
        for j in range(length[i]):
            current_line += 1
            if prediction[i, j] == 0:
                list_location[0].append(current_line)

            elif prediction[i, j] == 1:
                list_location[1].append(current_line)

            elif prediction[i, j] == 2:
                list_location[2].append(current_line)

            elif prediction[i, j] == 3:
                list_location[3].append(current_line)

            elif prediction[i, j] == 4:
                list_location[4].append(current_line)

            elif prediction[i, j] == 5:
                list_location[5].append(current_line)

            elif prediction[i, j] == 6:
                list_location[6].append(current_line)

            elif prediction[i, j] == 7:
                list_location[7].append(current_line)

            elif prediction[i, j] == 8:
                list_location[8].append(current_line)

        current_line += 1

    return list_location

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/post', methods=["GET", "POST"])
def test():
	if request.method == "POST":
		file = request.files['file'].read()
		temp_file_name = str(uuid.uuid4())+".txt"

		with open("./"+temp_file_name, "wb") as f:
			f.writelines(file)
			f.close()

		prepared_file = prepare_file(temp_file_name)
		inp = convert_file(prepared_file)
		#os.remove("./"+temp_file_name)
		os.remove("./"+prepared_file)

		pred, length = sess.run([model.prediction, model.length], {model.input_data: inp})

		return to_list(pred, length)

if __name__ == "__main__":
	print "Loading tensorflow stuff"

	model = Model()

	sess = tf.Session()
	sess.run(tf.initialize_all_variables())

	saver = tf.train.Saver()
	tf.reset_default_graph()
	saver.restore(sess, '/Users/Josh/Documents/legal_parsing_service/models/saved_model/model_max.ckpt')
	model_glove = pkl.load(open("./glovevec_model_311.pkl", 'rb'))

	print('Starting the API')
	app.run(debug=True)
