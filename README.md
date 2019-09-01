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
For the preprocessing of a gzipped sparse embedding:
	
	cd src
	python preprocess/preprocess_sparse_embedding --embedding <path_to_gzipped_embedding>

This will result in npz format embedding matrix and word indexing files in the `data` directory.
	
To preprocess a knowledge base (and its assertions) run:

	cd src
	python preprocess/preprocess_cnet --kb <path_to_json_assertion>
	
This will result in pickled concept indexing files and word-concept dictionaries in the `data` directory.
	
To preprocess [dense glove embeddings][4] run:

	cd src
	python preprocess/preprocess_dense_embedding --embedding <path_to_dense_embedding>

This will result in pickled embedding matrix and word indexing files in the `data` directory.	
	
To preprocess [contextual sparse embeddings][5] run:

	cd src
	python preprocess/preprocess_contextual --embedding <path_to_contextual_gzipped_embedding>

This will result in pickled embedding matrix and word indexing files in the `data` directory.	

## Assign knowledge base concepts to sparse embedding dimensions
Make sure knowledge base and sparse embedding are preprocessed.

	cd src
	
Filter word embedding to contain words that are also present in the vocabulary of the knowledge base:

	python sparse_alignments/filter_embedding.py --embedding <path_to_npz_embedding> --vocabulary <path_to_pickled_vocabulary>

Then based on the knowledge base present during filter, generate a word-concept matrix:

	python sparse_alignments/word_concept_matrix.py --embedding <path_to_filtred_npz_embedding>
	
For the (TAB and TAC) evaluations split train and test sets for each concept:

	python sparse_alignments/train_test_split.py --embedding <path_to_filtred_npz_embedding>

Compute meta-concepts based on word-concept matrix:

	python sparse_alignments/meta_concepts.py --word-concept <path_to_word_concept_matrix>
	
Run the alignment based on NPPMI:
	
	python sparse_alignments/alignment_NPPMI.py --embedding <path_to_filtered_npz_embedding>

The results can be found in the `results\nppmi` folder. The alignment for needed for evaluation will be in the `results\nppmi\max_concept\` folder.
	
	
Evaluate alignments. Input pickled files are in the `results/nppmi/max_concepts` folder.

	python sparse_alignments/evaluation.py --alignment <path_to_pickled_alignment>
	
## Assign knowledge base concepts to dense embedding dimensions	

Similarly to sparse embeddings run the following for dense embeddings:

	cd src
	python dense_alignments/filter_embedding.py --embedding <path_to_pickled_embedding> --vocabulary <path_to_pickled_vocabulary>
	python dense_alignments/word_concept_matrix.py --dense-matrix <path_to_filtred_npz_embedding>
	python dense_alignments/train_test_split.py --embedding <path_to_filtred_npz_embedding>
	python dense_alignments/meta_concepts.py --concept <path_to_word_concept_matrix>
	python dense_alignments/alignment_NPPMI.py --embedding <path_to_filtered_pickled_embedding>
	python sparse_alignments/evaluation.py --alignment <path_to_pickled_alignment>

## Contextual embeddings

Sparsified contextual embeddings are investigated following the work of [Ragnato et al (2017)][6]. The framework containing the training and evaluation data can be found [here][7]. Training and evaluation embeddings should be preprocessed.

To process the train and eval folders (`train_dir` contains the xml and key files, `eval_dir` contains the directories containing xml and key files):

	cd src
	python contextual/process_xml.py --trainEmbedding <path_to_preprocessed_train_embedding> --trainDir <train_dir> --evalEmbedding <path_to_preprocessed_eval_embedding> --evalDir <eval_dir>
	
Then, to compute inter-intra distance ratio:
	
	python contextual/inter-intra_distance.py --embedding <path_to_preprocessed_train_embedding>
	
	
[1]: http://rgai.inf.u-szeged.hu/~berend/interpretability/sparse_glove_extended/
[2]: http://rgai.inf.u-szeged.hu/~berend/interpretability/contextual/
[3]: https://drive.google.com/open?id=19APSLGWn1IGAaWkpg9x-PoJo-fHI0SvS
[4]: https://nlp.stanford.edu/projects/glove/
[5]: http://rgai.inf.u-szeged.hu/~berend/interpretability/contextual/
[6]: https://www.aclweb.org/anthology/E17-1010/
[7]: http://lcl.uniroma1.it/wsdeval/home
