from dataset import *
from malform import *
import sys, getopt
import json
# Takes a tuple of the form ({'(a': 0, '(b': 1, 'a)': 2, 'b)': 3, 'START': 4, 'END': 5}, ['a', 'b'])
# and outputs [('(a', 'a)'),...]
def make_parens(t):
	_, vocab = t
	parens = [( f"({c}", f"{c})") for c in vocab]
	return parens

def deform_dataset(args):

	corrupt_file_paths = [args["corpus"]["malformed_train_corpus_loc"], 
				  		  args["corpus"]["malformed_test_corpus_loc"], 
				  		  args["corpus"]["malformed_dev_corpus_loc"]]

	file_paths = [args["corpus"]["train_corpus_loc"], 
				  args["corpus"]["dev_corpus_loc"], 
				  args["corpus"]["test_corpus_loc"]]
	
	for i, file_path in enumerate(file_paths):
		with open(file_path, "r") as data_clean:
			lines = data_clean.readlines()
			tokenized_lines = list(map(lambda line: line.split(" "), lines))

			parens = make_parens(utils.get_vocab_of_bracket_types(args["language"]["bracket_types"]))

			with open(corrupt_file_paths[i], "w+") as corrupt_dataset:
				for tokenized_line in tokenized_lines:
					deformed_s = deform(s=tokenized_line[:-1], parens=parens)
					corrupt_dataset.write(" ".join(deformed_s) + tokenized_line[-1])
D=3
K=3
C=64
opts, args = getopt.getopt(sys.argv[1:],"D:K:C:",["D=","K=","C="])

print("args: " +str(args) + ", opts: " +str(opts))
for opt, arg in opts:
     if opt == '-D':
         print ('max depth ' +str(D) + ' of dyck language being overridden with D=' + str(arg))
         D = int(arg)
     elif opt == '-K':
         print ('bracket complexity ' +str(K) + ' of dyck language being overridden with K='+str(arg))
         K = int(arg)
     elif opt == '-C':
         print ('context window ' +str(C) + ' of dyck language being overridden with C='+str(arg))
         C = int(arg)
      
print("BEFORE: K: " + str(K) + ", D: " + str(D) + ", C: " + str(C))
args = {"language": {
          "bracket_types": K,
          "dev_max_length": C,
          "dev_max_stack_depth": D,
          "dev_min_length": 1,
          "dev_sample_count":  5000,
          "test_max_length": C,
          "test_max_stack_depth": D,
          "test_min_length": 1,
          "test_sample_count": 30000,
          "train_max_length": C,
          "train_max_stack_depth": D,
          "train_min_length": 1,
          "train_sample_count": 150000,
          "evaluate": False
     },

     "corpus": {
      "train_corpus_loc": "data/dyck-clean-train.txt",
      "dev_corpus_loc": "data/dyck-clean-dev.txt",
      "test_corpus_loc":  "data/dyck-clean-val.txt",

      "malformed_train_corpus_loc": "data/dyck-corrupted-train.txt",
      "malformed_test_corpus_loc": "data/dyck-corrupted-dev.txt",
      "malformed_dev_corpus_loc":  "data/dyck-corrupted-val.txt"
     }

}

# making this script idempotent
print("deleting old files...")
try:
    os.remove(args['corpus']['train_corpus_loc'])
    os.remove(args['corpus']['dev_corpus_loc'])
    os.remove(args['corpus']['test_corpus_loc'])
    os.remove(args['corpus']['malformed_train_corpus_loc'])
    os.remove(args['corpus']['malformed_train_corpus_loc'])
    os.remove(args['corpus']['malformed_train_corpus_loc'])
except OSError:
    pass
print("old files deleted")
create_clean_dataset(args)
deform_dataset(args)

print("K: " + str(K) + ", D: " + str(D) + ", C: " + str(C))
logger = json.dumps(args['language'])
with open("language_config.json", "w") as outfile:
    outfile.write(logger)





			





