# Interpretability of Word Embeddings
## Requirements
Python 3.6
## Install
Install the latest code from GitHub.

    git clone https://github.com/Zsaknadrag/interpretability-word-embedding
    cd interpretability-word-embedding
    pip install -r requirements.txt

Download the sparse [glove][1] and [contextual][2] (bert) embeddings and the [assertion][3] files of the knowledge bases.
## Preprocess

	cd src
	
For the preprocessing of a gzipped embedding:
	
	python ./preprocess/preprocess_sparse_embedding --embedding <path_to_gzipped_embedding>
	
To preprocess a knowledge base (and its assertions) run:

	python ./preprocess/preprocess_cnet --kb <path_to_json_assertion>
	
## Assign knowledge base concepts to embedding dimensions.
    cd src




[1]: http://rgai.inf.u-szeged.hu/~berend/interpretability/sparse_glove_extended/
[2]: http://rgai.inf.u-szeged.hu/~berend/interpretability/contextual/
[3]: https://drive.google.com/open?id=19APSLGWn1IGAaWkpg9x-PoJo-fHI0SvS
