'''
FastCGI server configuration.

Author:     Pontus Stenetorp
Version:    2011-12-08
'''

from os.path import dirname, join as path_join

MODEL_PATH_BY_TOKEN = {
        # Token to model, example:
        # 'news': path_join(dirname(__file__), 'news.model'),
        }

DEFAULT_TOKEN = 'default'
# There is a special 'default' that is used if no token is given, example:
# MODEL_PATH_BY_TOKEN[DEFAULT_TOKEN] = 'news'
