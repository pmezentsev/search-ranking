import json
import traceback
from flask import Flask, jsonify, request
from search_engine import SearchEngine
# from guppy import hpy
# import gc

def return_exception(func):
    def wrapper():
        try:
            return func()
        except Exception as e:
            exc = traceback.format_exc()
            print(exc)
            return jsonify(exc), 500

    wrapper.__name__ = func.__name__
    return wrapper


def create_app():
    # run flask service
    app = Flask(__name__)
    search_engine = SearchEngine()

    @app.route('/ping', methods=['GET'])
    def ping():
        return {'status': 'ok'}

    @app.route('/query', methods=['POST'])
    @return_exception
    def query():
        queries = json.loads(request.json)['queries']
        result = search_engine.query(queries)
        # print(hpy().heap())
        return jsonify(result)

    @app.route('/update_index', methods=['POST'])
    @return_exception
    def update_index():
        documents = json.loads(request.json)['documents']
        index_size = search_engine.update_index(documents)
        # gc.collect()
        # print(hpy().heap())
        return {'status': 'ok', 'index_size': index_size}

    @app.route('/score', methods=['POST'])
    @return_exception
    def score():
        request_dict = json.loads(request.json)
        doc = request_dict['doc']
        query = request_dict['query']
        score = search_engine.score(doc, query)
        # gc.collect()
        # print(hpy().heap())
        return {'score': score}

    return app


app = create_app()
