#!/usr/bin/env python

'''
FastCGI hooking requests into the classifier.

Depends on:

* plup - http://pypi.python.org/pypi/flup

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2011-09-13
'''

from cgi import parse_qs
from flup.server.fcgi import WSGIServer
from json import dumps as json_lib_dumps
from os.path import dirname, normpath, join as path_join
from sys import stderr

try:
    from cPickle import load as pickle_load
except ImportError:
    from pickle import load as pickle_load

# We will use these to hack into the model interface
from resources import Document, Sentence, Annotation
from fcgiconf import DEFAULT_TOKEN, MODEL_PATH_BY_TOKEN

### Constants
LOADED_MODEL_BY_TOKEN = {}
###

def _load_model(token):
    model_path = MODEL_PATH_BY_TOKEN[token]
    # Try if we have already loaded a model with the same path
    for other_token, other_model_path in MODEL_PATH_BY_TOKEN.iteritems():
        if (normpath(model_path) == normpath(other_model_path)
                and other_token in LOADED_MODEL_BY_TOKEN):
            LOADED_MODEL_BY_TOKEN[token] = LOADED_MODEL_BY_TOKEN[other_token]
            break
    else:
        # We fall back to loading the model
        with open(model_path, 'rb') as model_file:
            LOADED_MODEL_BY_TOKEN[token] = pickle_load(model_file)

def _json_dumps(dic):
    return json_lib_dumps(dic, indent=4)

def _serve(query):
    resp_dict = {}

    try:
        to_classify = set(query['classify'])
    except KeyError:
        resp_dict['error'] = 'noClassifyArgument'
        return resp_dict

    try:
        token = query['token'][0]
    except KeyError:
        # No token provided, do we have a default?
        if not DEFAULT_TOKEN in MODEL_PATH_BY_TOKEN:
            resp_dict['error'] = 'noTokenSpecifiedAndNoServerDefaultSet'
            return resp_dict
        token = DEFAULT_TOKEN

    model = LOADED_MODEL_BY_TOKEN[token]

    doc = Document('<fcgi>', [], [], '<fcgi>')
    for s in to_classify:
        doc.abstract.append(
            Sentence(s, [Annotation(0, len(s), None)]))

        res_dict = {}
        for sent in doc:
            for ann in sent:
                ann_text = sent.annotation_text(ann)
                res_dict[ann_text] = model.classify(doc, sent, ann, ranked=True)
        resp_dict['result'] = res_dict

    return resp_dict

def simsem_app(environ, start_response):
    query = parse_qs(environ['QUERY_STRING'])

    # Call the main server
    resp_dict = _serve(query)

    # We always respond with '200 OK' since the errors are in JSON
    start_response('200 OK', [('Content-Type', 'application/json'), ])
    yield _json_dumps(resp_dict)

if __name__ == '__main__':
    if not MODEL_PATH_BY_TOKEN:
        print >> stderr, "ERROR: no models specified by 'fcgiconf.py', exiting"
        exit(-1)

    # Pre-load the models since this takes some time
    for token in MODEL_PATH_BY_TOKEN:
        _load_model(token)
    # Then we are ready to serve requests
    WSGIServer(simsem_app).run()
