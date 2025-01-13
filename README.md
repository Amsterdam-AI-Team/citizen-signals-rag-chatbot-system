# Agentic RAG support for citizens reporting issues in Amsterdam

This repository contains an Agentic RAG (Retrieval Augmented Generation) System
developed for the City of Amsterdam and used to explore the use of this technology
for improving the handling of citizen reports.
The solution automates responses to provide fast, empathetic, and context-specific feedback,
reducing delays and dissatisfaction. By automating these cases, the workload for municipal staff is reduced,
allowing more focus on complex issues.

<p align="center" width="90%">
    <img width="90%" src=./media/demo-image-4.png alt="{{ include.description }}">
    <i> Example interaction with the system.
        When a melding is received, the system retrieves relevant information
        from various sources, with the aim of providing useful information
        about the issue, whether, when and how it would be resolved
        and what the citizen can expect. </i>
</p>


_**Disclaimer:**_ Due to the sensitive nature of some of the underlying data sources (e.g. the historical citizen reports),
running the system requires incorporating own data and making the necessary adjustments.
We hope, however, that its architecture, code and underlying prompts can serve as inspiration for similar use cases.


### Meldingen Background

Amsterdam's [Meldingen](https://meldingen.amsterdam.nl/) system is an online system
which allows citizens to report issues in public spaces.
Citizens can report issues suck as rubbish or a maintenance issue on the street
or in a park, a dangerous traffic situation or disturbance from people or cafe’s.
According to data from 2023, the City receives ± 400.000 citizen reports per year.

With 25-30% of reports requiring no (new) action from the municipality,
it is expected that automatically collecting information from internal systems
and providing the citizen with details about the issue would have a significant impact on users.
The solution would provide citizens with fast and accurate responses,
it will make the process of handling reports more transparent,
it will create user-friendly interaction, and - in the long term -
ensure better service for reports that do require human handling.

More information about the envisioned solution and its impact on users can be found in the full [report](OPENRESEARCH_LINK_TO_BE_ADDED).

## How it works

Our solution covers the most relevant components of an agentic RAG architecture.
It focuses primarily on the implementation of diverse tools which can help us assess
the feasibility of different scenarios and the adaptability and scalability of the system.


<p align="center" width="90%">
    <img width="90%" src=./media/Agentic-RAG-Meldingen-EN.png alt="{{ include.description }}">
    <br><i> Overview of our Agentic RAG architecture </i>
</p>


In the core of the system is the [central agent](./src/central_agentic_agent.py)
It serves as an orchestrator, which receives instructions regarding the task and the report,
collects all relevant details about a report, such as the type of problem and the location,
then independently determines which tool is suitable for gathering
information,

In it's [configuration](./src/config.py) , the agent receives instructions about its task,
different guidelines related to the resolution of reports, as well as communication rules.
Furthermore, the central agent is provided with an explanation of the [available tools](./src/tools)
and their functionalities to use them effectively for reports.
Finally, the central agent is instructed to follow a structured reasoning process,
documenting intermediate steps such as the report, selected tools, gathered information,
and preliminary conclusions, ensuring a comprehensive overview of actionable information.

The reasoning processes and effectively combining all gathered data is essentially
performed by an LLM. In most experiments we use own deployment of GPT4o, however,
it is possible to optimize the system for the use of
[any open-source model](https://github.com/Amsterdam-AI-Team/citizen-signals-rag-chatbot-system/tree/feat/SAI-2270-open-source-llm).
This does require adjusting the templates (especially the scratchpad) and configuration
(including some of the parameters in the langchain agent initialization)
to correspond to the desired model.


More information about the agent, as well as the individual tools can be found in the full [report](OPENRESEARCH_LINK_TO_BE_ADDED).

## Installation 

1) Clone this repository:

```bash
git clone https://github.com/Amsterdam-AI-Team/citizen-signals-rag-chatbot-system.git
```

2) Install pyaudio driver:

```bash
sudo apt-get install portaudio19-dev
``````

3) Install all dependencies:

```bash
pip install -r requirements.txt
```

The code has been tested with Python 3.10.0 on Linux.



## Usage

_**Disclaimer:**_  This repository is intended primarily for transparency and inspiration.
Due to the sensitive nature of some of the underlying data sources
(e.g. the historical citizen reports), running the system requires incorporating
own data and making the necessary adjustments.
*This could hinder the usage of the system outside of the secure municipal environment.*

### Configuration
Before use, please adjust all paths in the [config.py](./src/config.py) file.

You can also decide whether to `track_emissions`,
select a different `embedding_model_name` (used for the retrieval of historical reports),
decide on a model in the `model_dict`.

You can also adjust all agent templates according to own preferences.

### Secrets

Furthermore, you will need to add a `[my_secrets.py](./src/my_secrets.py) with an `API_KEYS` object containing
- `openai_azure` key: an OpenAI API key correspodning to the endpoint in the [config.py](./src/config.py)
- `co2-signal` key (if `track_emissions=True`)

### Option 1: Run via chat interface

You can use the chat interface by running the app located in the source directory using the following command:

```bash
python src/app.py
```
This will start the chatbot on your localhost, allowing you to interact with it via a web interface.
By default you can access the app locally on port 5000 ([127.0.0.1:5000](127.0.0.1:5000)).

### Option 2: Call the central agent

Alternatively, you can directly call the central agent by providing
the report, address and any additional information.
To do that, adjust the examples on the bottom of the [central_agentic_agent.py](./src/central_agentic_agent.py) script use the following command:

```bash
python src/central_agentic_agent.py
```

### Running via Azure ML

If you wish to run this code via Azure ML services, you can open the repository and run app.py in VS Desktop mode.
This will ensure the localhost application works properly.

#### Internal storage

For internal use of the original meldingen data, mount the corresponding storage container using:

```bash
sh blobfuse_meldingen.sh
```

#### TTS Note

The current implementation for reading messages out loud is not compatible with Azure OpenAI API because the tts model is not (yet) supported.


## Contributing

We welcome contributions! Feel free to [open an issue](https://github.com/Amsterdam-AI-Team/citizen-signals-rag-chatbot-system/issues), submit a [pull request](https://github.com/Amsterdam-AI-Team/citizen-signals-rag-chatbot-system/pulls), or [contact us](https://amsterdamintelligence.com/contact/) directly.

## Acknowledgements

This repository was created by [Amsterdam Intelligence](https://amsterdamintelligence.com/) for the City of Amsterdam.

## License 

This project is licensed under the terms of the European Union Public License 1.2 (EUPL-1.2).
