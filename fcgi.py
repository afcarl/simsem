#!/usr/bin/env python

'''
FastCGI hooking requests into the classifier.

Depends on:

* plup - http://pypi.python.org/pypi/flup

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2011-09-13
'''

from os.path import dirname, join as path_join
from json import dumps as json_lib_dumps
from flup.server.fcgi import WSGIServer
from cgi import parse_qs

try:
    from cPickle import load as pickle_load
except ImportError:
    from pickle import load as pickle_load

# We will use these to hack into the model interface
from resources import Document, Sentence, Annotation
#from expriment.common import cache_simstring

### Constants
# XXX: Ugly global
MODEL = None
MODEL_PATH = path_join(dirname(__file__), 'ge.model')
###

def _load_model():
    global MODEL
    with open(MODEL_PATH, 'rb') as model_file:
        MODEL = pickle_load(model_file)

def json_dumps(dic):
    return json_lib_dumps(dic, indent=4)

def simsem_app(environ, start_response):
    global MODEL

    resp_dict = {
            'error': 0,
            }

    query = parse_qs(environ['QUERY_STRING'])
    try:
        # Make sure we have unique strings to classify
        to_classify = set(query['classify'])

        if to_classify:
            doc = Document('<fcgi>', [], [], '<fcgi>')
            for s in to_classify:
                doc.abstract.append(
                        Sentence(s, [Annotation(0, len(s), None)]))
           
            # Cache for a potential speed-up later on
            #cache_simstring((doc, ), verbose=False)

            res_dict = {}
            for sent in doc:
                for ann in sent:
                    ann_text = sent.annotation_text(ann)
                    res_dict[ann_text] = MODEL.classify(doc, sent, ann, ranked=True)
            resp_dict['result'] = res_dict
        else:
            # Hack...
            raise KeyError
    except KeyError:
        resp_dict['error'] = 1
    
    start_response('200 OK', [('Content-Type', 'application/json')])
    return [json_dumps(resp_dict)]

if __name__ == '__main__':
    # Pre-load the model since this takes some time
    _load_model()
    # Then we are ready to serve requests
    WSGIServer(simsem_app).run()
