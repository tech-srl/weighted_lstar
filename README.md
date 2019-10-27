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

>1. Pytorch
>2. [Graphviz](http://graphviz.readthedocs.io/en/stable/manual.html#installation) (for drawing the extracted PDFAs). 
>3. [NumPy and SciPy](https://scipy.org/install.html) (for Scikit-Learn)
>4. [Scikit-Learn](http://scikit-learn.org/stable/install.html) (for the SVM classifier)
>5. [Matplotlib](https://matplotlib.org/users/installing.html) (for plots of our networks' loss during training)

If you are on a mac using Homebrew, then NumPy, SciPy, Scikit-Learn, Matplotlib, Graphviz and Jupyter should all hopefully 
work with `brew install numpy`, `brew install scipy`, etc. 

If you don't have Homebrew, or wherever `brew install` doesn't work, try `pip install` instead. 

For Graphviz you may first need to download and install the package yourself ([Graphviz](https://www.graphviz.org/download/)), 
after which you can run `pip install graphviz`. 
If you're lucky, `brew install graphviz` might take care of all of this for you by itself.


### Extracting from Existing Networks
You can apply the full example directly to any language model (eg RNN, Transformer, other..) that provides the following API:
>1. `TODO`

To apply only the weighted lstar extraction to a given language model, it needs only the subset: `TODO`.


### Citation
You can cite this work using:

TODO

### Credits
This repository contains a sample train file from the [SPiCe](https://spice.lis-lab.fr) (Sequence Prediction Challenge) competition, 
more samples can be obtained and played with on the website.
We also use Rémi Eyraud's NDCG evaluation code from the same challenge to compute NDCG on our extracted models, though we change it slightly to allow the use of different values of `k`.
The ALERGIA comparison in the paper was done using [FLEXFRINGE](https://automatonlearning.net/flexfringe/).
