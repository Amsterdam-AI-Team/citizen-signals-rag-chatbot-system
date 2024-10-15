# RAG Chatbot System for Citizen Signals

The Retrieval Augmented Generation (RAG) Chatbot System is designed for the Municipality of Amsterdam to efficiently handle citizen signals (meldingen). When a melding is received, the system retrieves relevant information using RAG, with the aim of resolving the issue before it is forwarded to the Municipality’s official Signals system. This approach improves the overall user experience by reducing the number of meldingen that need to be escalated, ensuring faster and more efficient responses.


https://github.com/user-attachments/assets/71d751a4-6eda-4b89-85cf-71f9f11b52b4


## Background

The RAG Chatbot System for Citizen Signals was developed to streamline the process of managing citizen reports (meldingen) for the Municipality of Amsterdam. By using Retrieval Augmented Generation (RAG), the system can pull in relevant information related to a melding and attempt to resolve it before escalating it to the Municipality's official Signals system. This proactive approach helps reduce the workload on the municipal team and enhances the user experience by providing quicker resolutions.

The system is designed to efficiently manage citizen-reported issues, improve response times, and minimize unnecessary entries into the Signals system. By integrating RAG, the chatbot retrieves contextual information and assists citizens directly, minimizing the need for manual handling. This innovative approach enhances overall efficiency and helps the municipality better serve its citizens.

## Folder Structure

* [`src`](./src): All source code files specific to this project.

## Installation 

1) Clone this repository:

```bash
git clone https://github.com/Amsterdam-AI-Team/citizen-signals-rag-chatbot-system.git
```

2) Install pyaudio driver:

```bash
sudo apt-get install portaudio19-dev
```

3) Install all dependencies:

```bash
pip install -r requirements.txt
```

The code has been tested with Python 3.10.0 on Linux/MacOS/Windows.

## Usage

### Step 1: Navigate to scripts

First, navigate to the source directory:

```bash
cd src
```

### Step 2: Run the Chatbot

You can run the chatbot locally by executing the following command:

```bash
python3 app.py
```

This will start the chatbot on your localhost, allowing you to interact with it via a web interface.

### Running via Azure ML

If you wish to run this code via Azure ML services, you should open repository and run app.py in VS Desktop mode. This will ensure the localhost application works properly.

### Notes

- You will need an OpenAI API key for answer generation and image processing. This API keys should also be specified in the configuration file. It is possible to use different LLMs of your choice, but doing so will require modifying the code accordingly. 

- Current implementation for reading messages out loud is not compatible with Azure OpenAI API because the tts model is not (yet) supported.


## Contributing

We welcome contributions! Feel free to [open an issue](https://github.com/Amsterdam-AI-Team/citizen-signals-rag-chatbot-system/issues), submit a [pull request](https://github.com/Amsterdam-AI-Team/citizen-signals-rag-chatbot-system/pulls), or [contact us](https://amsterdamintelligence.com/contact/) directly.

## Acknowledgements

This repository was created by [Amsterdam Intelligence](https://amsterdamintelligence.com/) for the City of Amsterdam.

Optional: add citation or references here.

## License 

This project is licensed under the terms of the European Union Public License 1.2 (EUPL-1.2).
