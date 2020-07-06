# Vocabularies
`python/vocab.py` should be used to generate vocab files. Generated ones are stored in the `vocabulary` directory.

## Creating a vocabulary file
`python/vocab.py` creates a vocabulary based on a pretrained word embeddings file (if given) and the words in the dataset. It creates an instantiation of a `Vocabulary` class which has functions to retrieve words by their IDs and vice versa. It also contains fields for vocabulary sizes that can then be used to initialize sizes of LSTM layers. Finally, the `Vocabulary` also contains a field, `self.word_embeddings`, that is directly a `torch.nn.Embedding` layer that can be used to encode word IDs in your model.

To create a vocabulary that will be saved to disk, call `vocab.py` with the following arguments:
* `--vector_filename`: path to the pretrained word embeddings file you want to use. As of now, this script can only parse embeddings in Glove pretrained embedding txt files. In the future, support for word2vec, fastText, and other pretrained embeddings files should be included. If left blank, the resulting vocabulary will return an `Embedding` layer that contains just the words in the training dataset and will be trained (i.e., `requires_grad=True`).
* `--embed_size`: size of the word embeddings. This should be set to the size of the pretrained embeddings you are using, or to a desired embedding size if not using pretrained embeddings.
* `--oov_as_unk`: by default, any words found in the dataset that do not contain pretrained word embeddings and also appear frequently enough in the dataset (over some threshold) are added as random vectors to the vocabulary. To disable this feature, and thus treat such words as `<unk>` tokens, enable this flag.
* `--lower`: lowercase the tokens in the dataset.
* `--keep_all_embeddings`: by default, words that are never seen in the dataset are not kept as word embeddings in the final vocabulary dictionary. This greatly speeds up processing time and reduces the size of the vocabulary that is needed in memory and on disk. For an interactive demo, however, the embeddings for other unseen words should be kept. To do so, enable this flag.
* `--train_embeddings`: by default, if a pretrained embeddings file has been loaded, the resulting `torch.nn.Embedding` layer that is initialized is frozen, i.e., `requires_grad=False`. To enable fine-tuning of the embedding layer, enable this flag.
* `--use_speaker_tokens`: by default, the vocabulary is initialized with generic start- and end-of-sentence tokens, i.e., `<s>` and `</s>`. In order to use speaker-specific tokens, e.g. `<architect>` and `</architect>`, enable this flag.
* `--threshold`: the rare word threshold, below which items in the dataset that do not have pretrained embeddings are treated as `<unk>`.
* `--verbose`: prints the entire dictionary of the vocabulary when initialization is finished.

The resulting vocabulary will be pickled and saved to `vocabulary/` and will have a filename that is a combination of the pretrained word embeddings filename and various parameters you have specified (e.g. lowercasing, rare word threshold, and use of speaker tokens). A log of the console messages printed during vocabulary creation is also saved to the same directory.

## Using a generated vocabulary file
The vocabulary can be easily loaded using `pickle.load()`. The `Vocabulary`'s `self.word_embeddings` field, which contains embeddings for all words in the entire dataset, should be directly used as a `torch.nn.Embedding` layer in your model. The embedding for the `<unk>` token is defined to be the average of all word embeddings for words seen in the training set.

Calling the vocabulary will allow you to look up a word ID for a given word using either the words-to-IDs dictionary based on the entire dataset or only that of the train set, the latter of which should be used for decoding. When calling the vocabulary for word lookup, lookups for input words should be simply called as `vocabulary(word)`, while lookups for output words should use `vocabulary(word, split='train')` to disallow unseen tokens from being generated at the output. **Additionally, the output size for your model's final `Linear` layer should use `vocabulary.num_train_tokens` in order for this word lookup to function correctly.**
