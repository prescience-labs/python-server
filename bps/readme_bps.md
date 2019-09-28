# README

Blake Perry Smith's (BPS) work on Data Intel. blakeperrysmith@gmail.com

---
## What's done

### `python-server/` `develop` branch
This branch has all the best and prettiest code of the following pipeline.

First, a bird's eye view:

`infer/` - The endpoint which takes a string of a single review's text and returns an analyzed sentiment_doc.

This draws on the following two modules to do the heavy lifting:

`auto_aspect.py` - Totally novel BPS code which looks at the sentence structure of the input string and returns a list of aspects / categories / n-grams of potential interest.

`inference.py` - Copied-over source code from Intel's NLP Architect ABSA which takes a list of aspects and the name of some stored opinion lexicon, finds sentiment events, and returns the sentiment_doc structure.

Now, in more detail:

#### `auto_aspect.py`

This is comprised of actually very few functions, but this was probably the hardest piece to write. Drawing on linguistic theory and lots of manual analysis, this basically takes the raw input string and looks for spacy-tagged dependency relations of a certain type. Then, there are a few helper functions to manipulate the data, including to group together any aspect candidates that happen to be neighbors into compound aspects. It seems to be working pretty well, but there is surely more to be done including appending to the stop words list and adding other means of extracting aspects as mentioned below.

Multi-method auto-aspect extraction would be even better, if more is better. The idea here is to do other things (I discuss these in detail under `what could be done`) in addition to these linguistically-informed techniques to catch other spans we might be interested in as aspects.

#### `inference.py`

The functions, classes, objects, and utilities this leverages were embedded in different places throughout the Intel NLP Architect codebase but should be searchable on [GitHub](https://github.com/NervanaSystems/nlp-architect) by function names etc as those were (I think entirely?) kept the same. This module is completely severed from Intel's actual `architect` python package and functions using only the root dependencies and in the isolated environment of the service. To emphasize, the pip-installable package from Intel is NOT a dependency of this module or any part of the service.

I've organized Intel's code into three main groups:
  1. `inference.py`
   - These are the core functions that perform the sentiment inference on the raw input data. They use the automatically generated aspect list and the hard-coded opinion lexicon to spin up the sentiment doc that is returned by the service. This would also probably be the place where you could tinker and add in functions to insert any extra fields in the data structure to be included in the JSON response.
  2. `utils.py`
    - These are the helper functions used by any of the Intel code, just thrown into a bucket. This also includes the main custom data types and objects Intel uses for its inference calculations.
  3. `bist/`
    - This directory houses all of the "stuff" needed for the preprocessing of the raw inference data so that the inference engine can run properly. It contains a pretrained `bist.model` which you can read about [here](http://nlp_architect.nervanasys.com/bist_parser.html?highlight=bist) which does dependency parsing in a fancy way which I couldn't figure out how to replicate in my own way. I don't really know what's going on with anything in this directory, but it works great and it's only dependencies outside of the standard library and helper functions included in `utils.py` are numpy, spacy, and dynet==2.1. Should probably freeze the version of numpy and spacy too.


Even more detail should be at the comment level including doc strings.

## What's unfinished

### `python-server/` `auto-aspect` branch

This has some notes and additional attempts at various ideas. The branch is much messier and was my sandbox for experimentation. Basically, here lies the early inklings & notes of my attempts at the ideas listed in the `what could be done` section below.

`python-server/app/modules/core/api.py`
- This is not yet hooked up to take an input string from the actual API request body. Currently a hard-coded dev string.
- Some kind of data validation would be good, here as well.

#### `train/`

I never got around to developing an endpoint which would take a large dataset such as the amazon datasets, and returns a "model". Mostly because there is not really such thing as training or a traditional model in the paradigm I ended up using. This endpoint or something like it could, I guess, be a replicability measure which recreates the environment needed for inference, in a way "training" it. This would entail things like downloading or reformatting the baseline opinion lexicon. In the future, components may very well need to be trained and this could be here obviously.

#### `update/`

This could be a good place to do upserts on the opinion_lex.

## What could be done

* Training a custom Named-Entity Recognition (NER) model using Spacy.
  * This NER model would run as part of the `auto_aspect` module
  * It would have to be trained on a few hundred (at least) hand-annotated examples of spans which contain the types of things we are interested in (product names, aspects, etc)
  * The process is documented by Spacy pretty well and is basically just adding more labels to one of the built-in NER models they provide
  * I would use Doccano for doing the annotations, which also nicely allows you to preserve other fields in your data structure containing your training texts by storing everything under the `meta` field. In addition to the annotated labels, all that's in meta will be saved when you export the JSON.
  * Then, get the output JSON of doccano into the format Spacy wants and follow the Spacy docs about training, testing, and saving the updated model.
* Derive and use some kind of baseline `aspect lexicon` for additional methods of automatically extracting aspects from novel text during the servicing of the `infer/` endpoint.
  * One way to do this might be running the current V1 of BPS' auto_aspect module on the amazon data set to get a big set of aspect candidates.
    * This is also a good way to check the robustness of V1 auto_aspect and tweak things accordingly
  * Once you have this set of aspect candidates, you could do a number of things to create reusable components which could be plugged into the actual module:
   * You could abstract away the surrounding contexts of the aspects found by the previous step to POS tags. (So, something like `DET ADJ ADJ <asp>NOUN NOUN</asp> DEM VERB PRON`) Then, you could see which are the most frequent POS contexts surrounding aspects. You could visually validate that the aspects occurring in those are the types of things we would consider quality aspects. Then, take the top-n most frequent POS patterns and add a function in auto_aspect which checks to see if any of the same patterns exist in the new review text. If so, the stuff between the `<asp></asp>` tags in the pattern is potentially a good aspect term.
   * ALSO, by deriving a big list of quality aspect terms, you could train an embedding model using [FastText](https://fasttext.cc/docs/en/python-module.html) and do some interesting vector similarity computations. I think it'd be important to train (or at least update/bootstrap from a pretrained) a new set of vectors based on the amazon dataset as a whole because our domain is pretty specific. So, say you now have a set of custom embeddings for the entire amazon dataset and a list of where aspects are in that dataset, you could compare nouns or noun-grams to those embeddings. I think fasttext has some kind of distance-between-two-words function.
* Derive a better opinion lexicon. The current one is directly from Intel's github for the project. We could do something like run the nlp architect on the amazon datasets or more reasonably some subset of it to get a better, more domain-specific set of opinion word entries.

---
### Final Notes

#### Colab

Google Colab notebooks of my experiments are exported and included as well as .ipynb files

#### Bookmarks & links

I'm also including the .html file of my Chrome bookmarks folder of any links relating to my work on Data Intel.

#### Thanks

Overall, this has been a great experience and I've learned a lot. Thanks for the opportunity and all the hard work you guys have put into this too. Can't wait to see what becomes of this all. With this I'm pretty much signing off for good in terms of any substantive contributions; but, I'm always happy to talk ideas and answer questions about anything I've written. Best ways to reach me are Slack, FB Messenger, and blakeperrysmith@gmail.com.
