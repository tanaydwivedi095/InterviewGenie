# InterviewGenie

**InterviewGenie** is an advanced interview bot designed to:

- Ask intelligent and relevant questions to candidates.
- Score candidates based on their responses.
- Answer questions posed by the interviewee.

This project leverages Retrieval-Augmented Generation (RAG) to create a dynamic and interactive interview process. It integrates multiple versions of trained models, scaling from smaller datasets to massive ones, ensuring adaptability and precision.

---

## Features

- **The model is able to answer the questions asked by the interviewee.**
- **The model is able to ask interviewee questions and assess the answer given by the interviewee uploaded in form of text or audio.**
- **Audio to Text Conversion:** Converts audio responses into text for scoring and evaluation.
- **Dynamic Questioning:** Generates interview questions tailored to the context.
- **Candidate Scoring:** Analyzes and scores candidate responses.
- **Interactive Q&A:** Answers candidate queries during the interview process.
- **Streamlit UI:** A user-friendly interface where users can drop questions and instantly receive answers.
- **Scalable Models:** Includes multiple model versions trained on datasets ranging from 23K tokens to 6.7M tokens.

### Upcoming Feature

- **3D Interviewer Avatar:** A 3D face representing an interviewer will be integrated for a more engaging and realistic experience.

---

## Note

1. If you want to increase the dataset, simply drop the PDF files into a folder named `Dataset`. The model will automatically process the new data.
2. Before running the pipeline, download the necessary embeddings, data, and the `Dataset` folder from [this Google Drive link](https://drive.google.com/drive/folders/1JvJk0zykBL1H_SAWPehvmcQBJ0U0gP9Z?usp=sharing).

---

## File Details

The repository contains the following files, representing different versions of the model:

1. **`InterviewBot_v1.ipynb`**: Model trained on 23K tokens.
2. **`InterviewBot_v2.ipynb`**: Model trained on 96K tokens.
3. **`InterviewBot_v3.ipynb`**: Model trained on 0.26M tokens.
4. **`InterviewBot_v4.ipynb`**: Model trained on 2.65M tokens.
5. **`InterviewBot_v5.ipynb`**: Model trained on 6.70M tokens.
6. **`ui.py`**: Streamlit UI for interacting with the bot.
7. **`Speech to Text.ipynb`**: Jupyter notebook for converting audio responses to text.

Each file includes the full training pipeline, evaluation, and deployment logic for the respective version.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tanaydwivedi095/InterviewGenie.git
   ```
2. Navigate to the project directory:
   ```bash
   cd InterviewGenie
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Streamlit UI

1. Run the Streamlit application:
   ```bash
   streamlit run ui.py
   ```
2. A browser window will open. Enter your question into the input field to receive an answer.

### Notebooks

1. Open the desired model file (e.g., `InterviewBot_v5.ipynb`) in Jupyter Notebook or Google Colab.
2. Execute the cells step-by-step to:
   - Load the pre-trained model.
   - Initialize the RAG pipeline.
   - Interact with the bot.
3. Customize the bot's behavior, dataset, or scoring mechanisms as needed.

### Audio to Text Conversion

1. Open the `Speech to Text.ipynb` file in Jupyter Notebook or Google Colab.
2. Upload an audio file containing the interviewee's response.
3. Run the notebook to convert the audio to text.
4. Use the converted text for scoring and evaluation.

---

## Libraries Used

The project utilizes the following key libraries:

- **Transformers:** `AutoModelForCausalLM`, `AutoTokenizer`, `BitsAndBytesConfig`
- **SentenceTransformer**
- **PyTorch** (`torch`)
- **NumPy**, **Pandas**
- **TQDM**: For progress tracking
- **FITZ**: PDF processing
- **Regular Expressions** (`re`)
- **Streamlit**: For building the UI

---

## Model Details

### Versions and Training Data:

1. **V1**: Trained on **23K tokens** - Basic functionality with limited context.
2. **V2**: Trained on **96K tokens** - Enhanced understanding and response generation.
3. **V3**: Trained on **0.26M tokens** - Intermediate-level performance.
4. **V4**: Trained on **2.65M tokens** - High-quality responses with broader context handling.
5. **V5**: Trained on **6.70M tokens** - Advanced capabilities, optimal for complex interviews.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your message here"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Open a Pull Request.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgements

Special thanks to the developers of the libraries and tools used in this project for enabling seamless integration and implementation of advanced AI capabilities.

