# RAG Chatbot System for Citizen Signals

Retrieval Augmented Generation (RAG) Chatbot System designed for the Municipality of Amsterdam to handle citizen signals (i.e., meldingen). Given a melding, this system aims to retrieve useful information using RAG to already resolve the melding before it is forwarded to the Municipality of Amsterdam Signals system. This way we can improve user experience and avoid meldingen unnecessarily enter the Signals system.


https://github.com/user-attachments/assets/71d751a4-6eda-4b89-85cf-71f9f11b52b4


## Background

The Amsterdam RAG Chatbot System is a project developed to assist visitors of the amsterdam.nl website in easily accessing information related to the municipality. The scope could be enhanced by utilizing more data sources. By integrating advanced AI technologies, such as Retrieval Augmented Generation (RAG), the system provides accurate and context-aware responses. The chatbot supports multiple input types, including text and images, and offers flexibility in response generation through different LLMs.

## Folder Structure

* [`src`](./src): All source code files specific to this project.

## Installation 

1) Clone this repository:

```bash
git clone https://github.com/Amsterdam-AI-Team/citizen-signals-rag-chatbot-system.git
```

2) Install all dependencies:

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

**Note**: You will need an OpenAI API key for answer generation and image processing. This API keys should also be specified in the configuration file. It is possible to use different LLMs of your choice, but doing so will require modifying the code accordingly. 

**Note** Only the melding category 'Zwerfvuil' is now supported. You can add more protocols in the processors.py file.

**Note** Retrieval using RAG is now static (i.e., we provide the same answer format to each zwerfvuil melding). A database and retrieval mechanism should be included to make this more dynamic such that it adheres to your wishes.


## Contributing

We welcome contributions! Feel free to [open an issue](https://github.com/Amsterdam-AI-Team/citizen-signals-rag-chatbot-system/issues), submit a [pull request](https://github.com/Amsterdam-AI-Team/citizen-signals-rag-chatbot-system/pulls), or [contact us](https://amsterdamintelligence.com/contact/) directly.

## Acknowledgements

This repository was created by [Amsterdam Intelligence](https://amsterdamintelligence.com/) for the City of Amsterdam.

Optional: add citation or references here.

## License 

This project is licensed under the terms of the European Union Public License 1.2 (EUPL-1.2).
