from queue import Queue, LifoQueue

from flask import Flask, render_template, request, redirect, make_response

from client_thread import ClientThread
from utils import decode_image, inner_content, error_handler, create_initial_agent_queue, read_experience_logs

app = Flask(__name__)

image_queues = dict()
action_queues = dict()
agent_queue = LifoQueue()
clients = dict()
USER_ID_COUNTER = 0


@app.route('/')
@error_handler
def home():
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        # Request for inner content has been redirected here
        return make_response('Something went wrong.', 400)
    return render_template('index.html')


@app.route('/welcome')
@error_handler
@inner_content
def welcome():
    return render_template('welcome.html')


@app.route('/thanks/')
@error_handler
@inner_content
def thanks():
    return render_template('thanks.html')


@app.route('/experiment')
@error_handler
@inner_content
def experiment():
    return render_template('experiment.html')


@app.route('/image', methods=['POST'])
def process_image():
    data = request.get_json()
    image_data = data['image']
    user_id = data['user_id']
    image = decode_image(image_data)
    image_queues[user_id].put(image)
    return ""


@app.route('/action', methods=['GET'])
@error_handler
@inner_content
def get_action():
    user_id = int(request.args['user_id'])
    next_action = action_queues[user_id].get()
    # TODO return json with song information as well as file name (for credits)
    return next_action


@app.route('/session')
@error_handler
def start():
    # TODO any issues with concurrency?
    global USER_ID_COUNTER
    new_id = USER_ID_COUNTER
    USER_ID_COUNTER += 1
    action_queues[new_id] = Queue()
    image_queues[new_id] = Queue()
    clients[new_id] = ClientThread(user_id=new_id, action_queue=action_queues[new_id], image_queue=image_queues[new_id],
                                   agent_queue=agent_queue)
    clients[new_id].start()
    return str(new_id)


@app.route('/stop', methods=['GET'])
@error_handler
@inner_content
def stop():
    user_id = int(request.args['user_id'])
    clients[user_id].halt_learning()
    return ""


@app.errorhandler(404)
@error_handler
def page_not_found(error):
    return redirect('/')


if __name__ == '__main__':
    create_initial_agent_queue(agent_queue)
    # read_experience_logs()
    app.run()
