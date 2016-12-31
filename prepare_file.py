import os, re, nltk, sys, uuid

def touch(fname, times=None):
    with open(fname, 'a'):
        os.utime(fname, times)

def prepare_file(file):
	token_counter = 0
	file_number = 0
	error = 0
	file_name = str(uuid.uuid4())+".txt"
	touch(file_name)

	with open("./"+file_name, "wb") as f:
		current_file = open("./"+file).read()
		sentences = nltk.sent_tokenize(current_file.decode('utf-8'))
		sentences = [nltk.word_tokenize(sent) for sent in sentences]

		tagged_sentences = []
		chinks = []
		items = []

		grammar = r"""
			NP:
				{<.*>+}          # Chunk everything
				}<VBD|IN>+{      # Chink sequences of VBD and IN
			"""

		for sentence in sentences:
			tagged = nltk.pos_tag(sentence)
			tagged_sentences.append((sentence, tagged))

		for item in tagged_sentences:
			cp = nltk.RegexpParser(grammar)
			result = cp.parse(item[1])
			chinks.append(result)

		for chink in chinks:
			items.append(nltk.chunk.util.tree2conlltags(chink))

		for item in items:
			f.write("\n")

			for index, token in enumerate(item):
				try:
					f.write(token[0].encode('utf-8') + " " + item[index][1] + " " + item[index][2]+ " O")
					f.write("\n")

				except:
					error += 1
					pass

	f.close()

	return file_name