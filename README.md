# Weighted L-star
Welcome to our public repository, implementing the learning algorithm from our NeurIPS 2019 paper, Extracting Automata from Recurrent Neural Networks Using Queries and Counterexamples.


### This Repository
Run the `full_example.py` code for a full example, which will:
1. Train an RNN on given samples (choose the argument `--spice-example` to run on problem set 0 from the SPiCe competition, or `--uhl-num` to select a UHL (options: 1,2,3).
2. Extract from that RNN to make a PDFA (using weighted Lstar), WFAs (using the spectral algorithm), and n-grams.
3. Evaluate the WER and NDCG against the RNN for each of the extracted models
4. Save the RNN and extracted models, print the training and extraction times and measure results, and draw the PDFAs, all in a new folder `results/[lang]_[timestamp]`.

Example runs:

```python3 full_example.py --spice-example```

```python3 full_example.py --uhl-number=2```

You can also set the parameters for all the extractions and measures, e.g.:

```python3 full_example.py --spice-example --RNNClass=GRU --nPS=100 --lstar-time-limit=50 --ngram-total-sample-length=10000 --ndcg-k=3```

More parameters are listed in `full_example.py`.

### Package Requirements
##### Full Install
Everything here is implemented in Python 3. To use these notebooks, you will also need to install:

>1. [Pytorch](https://pytorch.org)
>2. [Graphviz](http://graphviz.readthedocs.io/en/stable/manual.html#installation) (for drawing the extracted PDFAs)
>3. [NumPy and SciPy](https://scipy.org/install.html) (for Scikit-Learn)
>4. [Scikit-Learn](http://scikit-learn.org/stable/install.html) (for the SVM classifier)
>5. [Matplotlib](https://matplotlib.org/users/installing.html) (for plots of our networks' loss during training)

If you are on a mac using Homebrew, then NumPy, SciPy, Scikit-Learn, Matplotlib, Graphviz and Jupyter should all hopefully 
work with `brew install numpy`, `brew install scipy`, etc. 

If you don't have Homebrew, or wherever `brew install` doesn't work, try `pip install` instead. 

For Graphviz you may first need to download and install the package yourself ([Graphviz](https://www.graphviz.org/download/)), 
after which you can run `pip install graphviz`. 
If you're lucky, `brew install graphviz` might take care of all of this for you by itself.


### Extracting from Existing Models
You can apply the full example directly to any language model (eg RNN, Transformer, other..) that provides the following API:
>1. `input_alphabet,end_token,internal_alphabet`: attributes listing the possible input tokens and the end token
>2. `initial_state`: function getting the model's initial state
>3. `next_state`: function with two parameters: the model's current state `s1` and current input token `t`, that returns a new state `s2` without modifying `s1`.
>4. `state_probs_dist`: function with single parameter: a model state, that returns the state's next-token distribution in the order of its `internal_alphabet` attribute. (e.g., so that the probability of stopping after state `s` is `model.state_probs_dist(s)[model.internal_alphabet.index(model.end_token)]` ).
>5. `state_char_prob`: function with two parameters: a model state `s` and internal token `t`, equivalent to evaluating `model.state_probs_dist(s)[model.internal_alphabet.index(t)]`. (Here because for some language models, this function might have a faster implementation than actually calculating the entire distribution).

For efficiency, it is also possible to implement directly the functions `weight` and `weights_for_sequences_with_same_prefix` as described in `LanguageModel.py`, which may be faster and more accurate when done directly in the model (as opposed to through the LanguageModel wrapper, which will apply the `next_state` and `state_char_prob` functions several times to compute them).
In particular for the spectral extraction, implementing the function `weights_for_sequences_with_same_prefix` directly in the RNN using batching can speed up the reconstruction.


### Citation
You can cite this work using:

TODO

### Credits
This repository contains a sample train file from the [SPiCe](https://spice.lis-lab.fr) (Sequence Prediction Challenge) competition, 
more samples can be obtained and played with on the website.
We also use Rémi Eyraud's NDCG evaluation code from the same challenge to compute NDCG on our extracted models, though we change it slightly to allow the use of different values of `k`.
The ALERGIA comparison in the paper was done using [FLEXFRINGE](https://automatonlearning.net/flexfringe/).
