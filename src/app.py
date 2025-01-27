"""Main app for chatbot-like interaction"""
import os

from flask import (
    Flask,
    jsonify,
    render_template,
    request
)

import config as cfg
from helpers.melding_helpers import load_session, save_session
from process_melding import MeldingProcessor

app = Flask(__name__, template_folder="../templates", static_folder="../static")
"""
Flask application for handling chat queries, session management, and audio streaming.

This module sets up the routes and logic for:
- Serving the main interface (`index.html`)
- Handling chat queries and maintaining session state
"""
global chat_history, session_active, melding_attributes

# Global variable to manage session state
chat_history = []
session_active = False
melding_attributes = {}


@app.route("/")
def home():
    """
    Render the home page of the web application.

    This route serves the index.html file located in the templates folder.

    Returns:
        str: Rendered HTML of the index page.
    """
    return render_template("index.html")


@app.route("/query", methods=["POST"])
def handle_melding_query():
    global chat_history, session_active, melding_attributes
    data = request.json
    melding = data.get("message")
    model = data.get("model", "gpt-4o")  # Default to 'ChatGPT 4o'
    model_name = cfg.model_dict[model]
    image = data.get("image")

    if session_active:
        # Continue with the existing session
        if not chat_history:
            chat_history = load_session(cfg.SESSION_FILE)
            melding_attributes = load_session(cfg.ATTRIBUTES_FILE)
    else:
        # Start a new session by clearing chat history
        chat_history = []
        melding_attributes = {}
        session_active = True  # Enable session for follow-up conversation

    # Initialize or update the complaint session
    melding_processor = MeldingProcessor(
        melding=melding,
        model_name=model_name,
        base64_image=image,
        chat_history=chat_history,
        melding_attributes=melding_attributes,
    )
    melding_processor.process_melding()

    # Update the chat history and save the session
    chat_history = melding_processor.chat_history
    melding_attributes = melding_processor.melding_attributes
    save_session(chat_history, cfg.SESSION_FILE)
    save_session(melding_attributes, cfg.ATTRIBUTES_FILE)

    # Get the response text from the last processed query
    response_text = chat_history[-1]["antwoord"]

    # Return response text to braodcast to frontend
    return jsonify({"response": response_text})


@app.route("/new_session", methods=["POST"])
def new_session():
    """
    Start a new chat session.

    This route clears the chat history, resets the session state, and removes any existing session file.

    Returns:
        Response: JSON object indicating that a new session has started.
    """
    global chat_history, session_active, melding_attributes

    # Clear chat history and reset session state
    chat_history = []
    melding_attributes = {}
    session_active = False

    # Remove the session file if it exists
    if os.path.exists(cfg.SESSION_FILE):
        os.remove(cfg.SESSION_FILE)
    if os.path.exists(cfg.ATTRIBUTES_FILE):
        os.remove(cfg.ATTRIBUTES_FILE)

    return jsonify({"response": "New session started. You can now ask your questions."})


@app.route("/clear-session", methods=["POST"])
def clear_session():
    """Deletes the session.json file if it exists."""
    try:
        if os.path.exists(cfg.SESSION_FILE):
            os.remove(cfg.SESSION_FILE)
        if os.path.exists(cfg.ATTRIBUTES_FILE):
            os.remove(cfg.ATTRIBUTES_FILE)
            return (
                jsonify({"status": "success", "message": "Session file deleted."}),
                200,
            )
        else:
            return (
                jsonify({"status": "success", "message": "No session file to delete."}),
                200,
            )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/stop_audio", methods=["POST"])
def stop_audio():
    """
    Stop the ongoing audio streaming.

    This route sets a flag to stop the current audio stream initiated by the read_aloud route.

    Returns:
        Response: JSON object confirming that audio streaming has been stopped.
    """
    global stop_flag
    stop_flag.set()  # Set the flag to stop the streaming
    return jsonify({"response": "Audio streaming stopped."})


if __name__ == "__main__":
    """
    Run the Flask development server.

    If this script is executed directly, it will start the Flask application
    in debug mode on the local development server at port 5001.
    """
    app.run(debug=True, port=5000)
