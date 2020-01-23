from flask import Flask, jsonify, request
from gensim.summarization.summarizer import summarize


app = Flask(__name__)


def summary_extraction(data):
    url = data['source_url']
    process_id = data['process_id']
    transcript_path = data['diarized_data']['response']['results']
    transcript_text = []
    for i in range(len(transcript_path)):
        try:
            transcript_text.append(transcript_path[i]['alternatives'][0]['transcript'])
        except:
            print([i, url])
    joined_transcript_text = " ".join(transcript_text)
    summary = summarize(joined_transcript_text)
    return url, joined_transcript_text, summary, process_id


def map_summaries(summary_list):
    mapped_summaries = []
    for summary_data in summary_list:
        mapped_summaries.append({
            'process_id':summary_data[3],
            'source_url': summary_data[0],
            'text': summary_data[1],
            'summary': summary_data[2]})
    return mapped_summaries


def consolidated_summary(json_file_data):
    data_ = json_file_data
    ls = []
    for i in range(len(data_['data'])):
        try:
            ls.append(summary_extraction(data_['data'][i]))
        except:
            print(i)
    # send back the json object for summary
    mapped_json_response = map_summaries(ls)
    return mapped_json_response


@app.route('/summarize', methods=['POST'])
def summarize_text():
    json_data = request.get_json()
    results_ls = {}
    results_ls['data'] = consolidated_summary(json_data)
    return results_ls


app.run(debug=True)
