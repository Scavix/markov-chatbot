from flask import Flask, request, jsonify, render_template
from markov import ContextualMarkovChatbot

app = Flask(__name__)

DATASETS = {
    "casual": "dialogs.txt",
    "bible": "bible.txt",
    "eminem": "eminem.txt"
}

order = 2
response_length = 20
context_size = 2

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    global bot
    data = request.json
    user_message = data["message"]
    dataset = data["dataset"]
    data_file = DATASETS.get(dataset, "dialogs.txt")
    order = int(data["order"])
    response_length = int(data["length"])
    context_size = int(data["context"])

    bot = ContextualMarkovChatbot(order=order, context_size=context_size)
    bot.train(data_file)

    bot_response = bot.generate_response(user_message, max_words=response_length)
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(debug=True)
