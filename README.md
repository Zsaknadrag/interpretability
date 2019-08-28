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
	
## Assign knowledge base concepts to sparse embedding dimensions.
Make sure knowledge base and sparse embedding are preprocessed.

	cd src
	
Filter word embedding to contain words that are also present in the vocabulary of the knowledge base:

	python ./alignments/filter_embedding --embedding <path_to_npz_embedding> --vocabulary <path_to_pickled_vocabulary>

Based on the knowledge base present druing filter, generate a word-concept matrix:

	python ./alignments/word_concept_matrix --embedding <path_to_filtred_npz_embedding>
	
[1]: http://rgai.inf.u-szeged.hu/~berend/interpretability/sparse_glove_extended/
[2]: http://rgai.inf.u-szeged.hu/~berend/interpretability/contextual/
[3]: https://drive.google.com/open?id=19APSLGWn1IGAaWkpg9x-PoJo-fHI0SvS
