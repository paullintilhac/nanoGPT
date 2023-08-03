from dataset import *
from malform import *

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

args = {"language": {
          "bracket_types": 2,
          "dev_max_length": 180,
          "dev_max_stack_depth": 5,
          "dev_min_length": 1,
          "dev_sample_count":  5000,
          "test_max_length": 180,
          "test_max_stack_depth": 5,
          "test_min_length": 1,
          "test_sample_count": 30000,
          "train_max_length": 180,
          "train_max_stack_depth": 5,
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

create_clean_dataset(args)
deform_dataset(args)






			





