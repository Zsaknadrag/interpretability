# Interpretability of Word Embeddings
## Requirements
Python 3.6
## Install
Install the latest code from GitHub.

    git clone https://github.com/Zsaknadrag/interpretability
    cd interpretability-word-embedding
    pip install -r requirements.txt

Download the sparse [glove][1] and [contextual][2] (bert) embeddings and the [assertion][3] files of the knowledge bases.
## Preprocess
For the preprocessing of a gzipped embedding:
	
	cd src
	python ./preprocess/preprocess_sparse_embedding --embedding <path_to_gzipped_embedding>
	
To preprocess a knowledge base (and its assertions) run:

	cd src
	python ./preprocess/preprocess_cnet --kb <path_to_json_assertion>
	
## Assign knowledge base concepts to sparse embedding dimensions
Make sure knowledge base and sparse embedding are preprocessed.

	cd src
	
Filter word embedding to contain words that are also present in the vocabulary of the knowledge base:

	python ./alignments/filter_embedding.py --embedding <path_to_npz_embedding> --vocabulary <path_to_pickled_vocabulary>

Then based on the knowledge base present during filter, generate a word-concept matrix:

	python ./alignments/word_concept_matrix.py --embedding <path_to_filtred_npz_embedding>
	
For the (TAB and TAC) evaluations split train and test sets for each concept:

	python ./alignments/train_test_split.py --embedding <path_to_filtred_npz_embedding>

Compute meta-concepts based on word-concept matrix:

	python ./alignments/meta_concepts.py --concepts <path_to_word_concept_matrix>
	
Run the alignment based on NPPMI:
	
	python ./alignments/alignment_NPPMI.py --embedding <path_to_filtered_npz_embedding>

Evaluate alignments (which align meta-concepts to bases):

	pythn ./alignments/evaluation.py --alingment <path_to_pickled_alignment>
	
## Assign knowledge base concepts to dense embedding dimensions	

## Contextual embeddings

	
[1]: http://rgai.inf.u-szeged.hu/~berend/interpretability/sparse_glove_extended/
[2]: http://rgai.inf.u-szeged.hu/~berend/interpretability/contextual/
[3]: https://drive.google.com/open?id=19APSLGWn1IGAaWkpg9x-PoJo-fHI0SvS
