from subprocess import Popen, PIPE, STDOUT

p = Popen(['python', './test_model.py'], stdout=PIPE, stdin=PIPE, stderr=PIPE)
stdout_data = p.communicate(input='/Users/Josh/Documents/legal_parsing_service/models/ner-lstm/embeddings/tag_b_embed')[0]
print stdout_data