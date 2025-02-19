# InterviewGenie

## Overview

InterviewGenie is an AI-powered interview assistant capable of acting as both an **Interviewee** and an **Interviewer**. It leverages Retrieval-Augmented Generation (RAG) to provide intelligent responses and questions, making it a powerful tool for candidates and interviewers alike.

## Features

### 1. **Interviewee Mode**

In this mode, the user provides questions, and InterviewGenie answers them using a RAG pipeline trained on multiple datasets. The model has been trained in five versions:

- **Version 1:** 23K tokens
- **Version 2:** 96K tokens
- **Version 3:** 0.26M tokens
- **Version 4:** 2.76M tokens
- **Version 5:** 6.8M tokens

### 2. **Interviewer Mode**

In this mode, InterviewGenie generates interview questions and evaluates responses. There are two models available:

#### **Model 1: Free Model**

- Uses `google/gemma-2b-it`
- Generates a set of **10 to 15** questions
- Allows the user to answer questions during the session
- At the end of the interview, displays all the answers in one place for review

#### **Model 2: Paid Model**

- Uses `OpenAI/o1-mini`
- Provides additional functionalities:
  - Users can select the **difficulty level** of interview questions
  - The model **scores responses** out of 10
  - At the end of the interview, it provides an **aggregate average score**
  - **Future Feature:** Personalized feedback to help users improve performance

### 3. **Streamlit Application**

InterviewGenie includes a Streamlit application (`gui.py`) that allows consumers to easily interact with the model. The application provides an intuitive graphical interface for both the Interviewee and Interviewer modes, enhancing user experience.

#### How It Works:

1. **Interviewee Mode:**
   - Users can input their questions through the GUI.
   - The application processes the question using the RAG pipeline and generates a context-aware answer based on trained embeddings and models.

2. **Interviewer Mode:**
   - Users can choose between the Free or Paid model:
     - **Free Model:** Generates 10-15 questions using `google/gemma-2b-it`.
       - Lets users answer questions during the session.
       - Displays all answers at the end of the interview for review.
     - **Paid Model:** Users can input an API key for enhanced functionality with `OpenAI/o1-mini`:
       - Select question difficulty using a slider.
       - Input answers for each question and receive a score out of 10.
       - Get an aggregate average score at the end of the session.
       - View all answers and their respective scores.

3. **Session Management:**
   - Tracks user progress, including current question index, answers, and scores.
   - Provides feedback and results after the session ends.

## Training Your Own RAG Pipeline

To train your own RAG pipeline:

1. **Prepare Your Dataset:**
   - Create a folder named `Dataset` in the project directory.
   - Add all PDF files containing training data to the `Dataset` folder.

2. **Run the Training Notebook:**
   - Open the `InterviewBot_v5.ipynb` notebook in Jupyter Notebook or JupyterLab.
   - Follow these steps in the notebook:
     - Load and preprocess the PDF data into embeddings.
     - Use the RAG pipeline to index the dataset for efficient retrieval.
     - Fine-tune the large language model (LLM) using HuggingFace's `transformers` library.
     - Generate and evaluate answers by augmenting prompts for better LLM instruction.

3. **Retrieve Top Contexts:**
   - The notebook helps retrieve the top-k relevant contexts from the dataset for any query.

4. **Generate and Validate Responses:**
   - Use the trained RAG pipeline to respond intelligently to user queries by generating context-aware responses.

## Required Files

Before using the model, download the required files for embeddings and data from the link below:

- [embeddings_vs.csv and data.csv](https://drive.google.com/file/d/1i_PkHlaUCbm5_ZafpTgFOHwdv9AITJ3S/view?usp=sharing)

Place these files in the project directory to ensure proper functioning of the InterviewGenie application.

## Running the Streamlit Application

To run the `gui.py` Streamlit application, follow these steps:

1. Open a terminal or command prompt.
2. Navigate to the project directory:
   ```sh
   cd InterviewGenie
   ```
3. Run the Streamlit application:
   ```sh
   streamlit run gui.py
   ```
4. Access the application in your browser using the link provided by Streamlit (usually `http://localhost:8501`).

## Code Flow Explanation

The following section explains the purpose of each code file and how they work together:

### 1. **gui.py**
   - The main Streamlit application that allows users to interact with the system.
   - Users can switch between Interviewee and Interviewer modes.
   - Handles user input and displays generated questions or answers.
   - Relies on the `Interviewer` and `Interviewee` classes for backend logic.

### 2. **Interviewer.py**
   - Implements the `Interviewer` class, which handles the generation of questions.
   - Supports both Free (Gemma-based) and Paid (OpenAI-based) question generation.
   - Scoring functionality evaluates user answers based on prompts sent to OpenAI’s LLMs.

### 3. **Interviewee.py**
   - Implements the `Interviewee` class, which retrieves answers based on user queries.
   - Uses pre-trained embeddings and a transformer-based language model (`google/gemma-2b-it`) to find relevant context and generate answers.
   - Includes functionality for cleaning and formatting LLM-generated responses.

### 4. **OpenAI_QuestionGenerator.py**
   - Implements a question generator using OpenAI’s APIs.
   - Allows users to define the difficulty level of the questions.
   - Processes LLM outputs to structure questions in a specific format.

### 5. **Gemma_QuestionGenerator.py**
   - Implements a free question generator using `google/gemma-2b-it`.
   - Generates questions in various categories such as definitions, formulas, scenarios, and case studies.
   - Allows for the customization of generated questions by setting parameters like the number of questions and generation temperature.
   - Supports preprocessing of generated questions to remove extra formatting and inconsistencies.

### 6. **Notebooks (e.g., InterviewBot_v5.ipynb, QuestionGenerator_v1.ipynb)**
   - Jupyter notebooks are used for training and fine-tuning the RAG pipeline.
   - Steps include data preprocessing, embedding creation, and LLM prompt augmentation.
   - `InterviewBot_v5.ipynb` is the most recent version, incorporating improvements in context retrieval and answer generation.

### 7. **Script_RAG_v1.ipynb**
   - Focused on implementing the RAG pipeline with a script-based approach.
   - Details the embedding retrieval process and query augmentation using the HuggingFace transformers library.

### 8. **Speech_to_Text_v1.ipynb**
   - Implements a speech-to-text pipeline for converting verbal queries into text.
   - This can be integrated with the InterviewGenie GUI for an enhanced user experience.

### File Relationships
   - `gui.py` interacts with both `Interviewer.py` and `Interviewee.py` to handle user requests.
   - `Interviewer.py` relies on `OpenAI_QuestionGenerator.py` or `Gemma_QuestionGenerator.py` for question creation.
   - `Interviewee.py` processes queries and utilizes pre-computed embeddings and transformer models for context-aware answers.
   - The notebooks (`InterviewBot_v5.ipynb`, etc.) are used to train and prepare the underlying models and embeddings required for `Interviewee.py` and `Interviewer.py`.

## Future Enhancements

- Personalized feedback based on interview responses
- Expanded dataset training for more accurate answers
- Support for additional question types and domains
- Enable users to answer questions by speaking their responses instead of typing

## Installation

### Clone the repository:

```sh
git clone https://github.com/tanaydwivedi095/InterviewGenie.git
```

### Navigate to the project directory:

```sh
cd InterviewGenie
```

### Install the required dependencies:

```sh
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

MIT License

