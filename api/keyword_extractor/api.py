from flask import Blueprint

summarizer_api = Blueprint('summarizer_api', __name__)


@summarizer_api.route('/summarize')
def get_summary():
    return {'summary': "this is the summary"}