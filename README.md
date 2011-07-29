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

## Resources ##

SimSem uses a large collection of lexical resources, the conversion and
processing scripts for these resources can be found under
`data/simstring/res/`. Since the resources are rather large they are
distributed separately and can be downloaded [here][res_main]
([mirror][res_mirror]).

<!-- Link collection -->
[bibtex]: http://en.wikipedia.org/wiki/BibTeX "BibTeX Entry on Wikipedia"
[iscl]: http://www.opensource.org/licenses/isc-license.txt "ISC License on opensource.org"
[liblinear]: http://www.csie.ntu.edu.tw/~cjlin/liblinear/ "LibLinear Homepage"
[simstring]: http://www.chokkan.org/software/simstring/index.html.en "SimString Homepage"
[res_main]: http://www-tsujii.is.s.u-tokyo.ac.jp/~pontus/share/simsem/simsem_lexical_resources_2011-07-27T1054Z.tar.gz "Lexical Resources"
[res_mirror]: http://udon.stacken.kth.se/~ninjin/share/simsem/simsem_lexical_resources_2011-07-27T1054Z.tar.gz "Lexical Resources Mirror"

<!-- "It's a trap!" (for bots) -->
[](http://bob.llamaslayers.net/contact.php?view=862)
