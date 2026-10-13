# Transformer implementation

This step implements the network and example preparation, not the full training
runner, synthetic query curriculum, final scorer, or a trained football model.

## Data flow

Text -> character token IDs -> token and position embeddings -> two causal
Transformer blocks -> final normalization -> scores over the vocabulary.

The default configuration uses width 128, four attention heads, feed-forward
width 512, dropout 0.1, and a context limit of 2048. With the pilot vocabulary of
127 IDs, the network has 691,456 parameters. All weights are initialized locally;
no pretrained weights are downloaded. Character IDs are arbitrary indices, not
numeric facts the network directly reasons over.

Each block applies pre-normalized attention and a feed-forward network, with
residual connections. Attention can use only current and earlier positions.
Padding is at the right edge and cannot be attended to as a key.

## Training examples

`corpus_example` adds BOS and EOS and shifts targets one position. Given
`BOS a b`, target characters are `a b EOS`. Every real next-character prediction
contributes to loss. The input remains different across the four representations.

`answer_example` combines a prompt and answer into the teacher-forced training
sequence, but masks targets belonging to the prompt. The final prompt position
predicts the first answer character. Answer characters and EOS contribute to loss.
Use this for the planned synthetic task training, not for injecting real frozen
evaluation answers into training.

`collate_examples` creates batches with right padding and ignored label value
`-100`. The model computes cross-entropy against the already-shifted labels.
An optimizer is required to update weights; computing loss alone does not train.

## Generation

`generate` accepts a single unpadded prompt, selects the highest-scoring next
token, appends it, and repeats. It stops at EOS, a token budget, or the context
limit, and reports why. It does not silently slide/truncate context. Padding,
BOS and unknown tokens are suppressed. The caller must preserve the stop reason
in evaluation outputs, especially if an answer is incomplete.

## Verification

Run from the project root:

    .venv\Scripts\python.exe -m unittest discover -s tests -v

Tests cover future-token isolation, padding invariance, answer-loss boundaries,
next-character shifting, finite gradients and loss reduction on a tiny toy sample,
EOS stopping and context validation. Toy optimization demonstrates working learning
machinery; it does not demonstrate recall of the football facts or compare formats.

The project-local environment currently uses a CPU PyTorch wheel for these checks.
It cannot use the NVIDIA GPU until a compatible CUDA wheel is installed.
