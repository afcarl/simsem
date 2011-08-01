# SimSem #

## Introduction ##

SimSem is a tool for semantic disambiguation using approximate string matching
and is distributed under the restrictions of the [ISC License][iscl].
To accomplish this SimSem uses large collections of strings such as
dictionaries, [LibLinear][liblinear] as its machine-learning component and
[SimString][simstring] for fast approximate string matching. Please see
the publication mentioned below for details.

If you draw inspiration from or base your work on SimSem, please cite the
below which is provided in [BibTeX][bibtex] format:

    @InProceedings{stenetorp2011simsem,
      author    = {Stenetorp, Pontus and Pyysalo, Sampo and Tsujii, Jun'ichi},
      title     = {SimSem: Fast Approximate String Matching 
          in Relation to Semantic Category Disambiguation},
      booktitle = {Proceedings of BioNLP 2011 Workshop},
      month     = {June},
      year      = {2011},
      address   = {Portland, Oregon, USA},
      publisher = {Association for Computational Linguistics},
      pages     = {136--145},
      url       = {http://www.aclweb.org/anthology/W11-0218}
    }

## Experiments ##

Clone this repository, then run the preparation script to download the lexical
resources, create the databases,to do some code generation (ugly) and build
external dependencies:

    ./prepare.sh

Experiments are then run using `test.py`, use the `-h` flag for more
information. For example, to replicate the main experiment (and plots) from
the BioNLP 2011 publication (use the tag `bionlp_2011`) you would run:

    mkdir bionlp_2011
    ./test.py -v -c INTERNAL -c INTERNAL-SIMSTRING -c INTERNAL-GAZETTER \
        -d BioNLP-ST-2011-Epi_and_PTM -d BioNLP-ST-2011-Infectious_Diseases \
        -d BioNLP-ST-2011-genia -d CALBC_II -d NLPBA -d SUPER_GREC \
        bionlp_2011 learning
    ./test.py bionlp_2011 plot

## Resources ##

SimSem uses a large collection of lexical resources, the conversion and
processing scripts for these resources can be found under
`data/simstring/res/`. Since the resources are rather large they are
distributed separately and can be downloaded [here][res_main]
([mirror][res_mirror]).

There is also data sets available in the
[BioNLP 2009 Shared Task format][bionlp_2009_st], two of which has been
converted from other formats. All data sets also have been sentence split and
tokenised in accordance to the
[BioNLP Shared Task 2011 pipeline][bionlp_st_2011_supporting]. These resources
can are further described on [the project wiki][datasets] be found under
`data/corpora` and you may also be interested in `prepare.sh` which
pre-processes and corrects some aspects of the data.

<!-- Link collection -->
[bibtex]: http://en.wikipedia.org/wiki/BibTeX "BibTeX Entry on Wikipedia"
[bionlp_2009_st]: http://www-tsujii.is.s.u-tokyo.ac.jp/GENIA/SharedTask/ "BioNLP 2009 Shared Task Homepage"
[bionlp_st_2011_supporting]: https://github.com/ninjin/bionlp_st_2011_supporting "BioNLP Shared Task 2011 Supporting Resources Preprocessing"
[datasets]: https://github.com/ninjin/simsem/wiki/Data-sets "Data Sets on the Wiki"
[iscl]: http://www.opensource.org/licenses/isc-license.txt "ISC License on opensource.org"
[liblinear]: http://www.csie.ntu.edu.tw/~cjlin/liblinear/ "LibLinear Homepage"
[simstring]: http://www.chokkan.org/software/simstring/index.html.en "SimString Homepage"
[res_main]: http://www-tsujii.is.s.u-tokyo.ac.jp/~pontus/share/simsem/simsem_lexical_resources_2011-07-27T1054Z.tar.gz "Lexical Resources"
[res_mirror]: http://udon.stacken.kth.se/~ninjin/share/simsem/simsem_lexical_resources_2011-07-27T1054Z.tar.gz "Lexical Resources Mirror"

<!-- "It's a trap!" (for bots) -->
[](http://bob.llamaslayers.net/contact.php?view=862)
