import time
import threading
from flask import Flask, jsonify, make_response, request as flask_request
from api_utils import BasmatinetPrediction
from queue import Empty, Queue


BATCH_SIZE = 24
BATCH_TIMEOUT = 0.5
CHECK_INTERVAL = 0.01

predictor = BasmatinetPrediction()
requests_queue = Queue()

app = Flask(__name__)


def worker():
    while True:
        requests_batch = []
        while not (
            (len(requests_batch) > BATCH_SIZE) or
            (len(requests_batch) > 0 and time.time() -
             requests_batch[0]['time'] > BATCH_TIMEOUT)
        ):
            try:
                requests_batch.append(
                    requests_queue.get(timeout=CHECK_INTERVAL))
            except Empty:
                continue
        batch_responses = predictor.inference_pipeline(requests_batch)


# Turn-on the worker thread.
threading.Thread(target=worker).start()


@app.route('/serving/predict', methods=['POST'])
def prediction_pipeline():
    # Get the image in base 64 and decode it
    payload = flask_request.form.to_dict(flat=False)
    image_b64 = payload['image'][0]
    # Put individual request in the requests_queue
    request = {'input': image_b64, 'time': time.time()}
    requests_queue.put(request)
    # Check if the response is available and serve it
    response = {'toto': 0.1}
    return make_response(jsonify(response))


@app.route('/serving/healthcheck', methods=['GET'])
def healthcheck():
    return 200


if __name__ == '__main__':
    app.run()
