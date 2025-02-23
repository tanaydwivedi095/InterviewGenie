import time
import streamlit as st
from Interviewee import Interviewee
from Interviewer import Interviewer

if __name__ == "__main__":
    st.title("InterviewGenie")
    name = st.text_input("Please, enter your name: ")

    # Initialize session state variables
    if "current_question_index" not in st.session_state:
        st.session_state.current_question_index = 0
    if "answers" not in st.session_state:
        st.session_state.answers = []
    if "scores" not in st.session_state:
        st.session_state.scores = []

    if name:
        st.write(f"Welcome, {name}!")
        option = st.radio("Do you want the bot to act as:", ["Interviewer", "Interviewee"])

        if option == "Interviewee":
            st.write("Enter the question that you want to ask:")
            query = st.text_input("Query")
            if query:
                interviewee = Interviewee(name)
                embeddings = interviewee.load_embeddings()
                data = interviewee.load_data()
                embedding_model = interviewee.load_embedding_model()
                tokenizer = interviewee.load_tokenizer()

                query_embedding = interviewee.embed_query(query, embedding_model)
                context = interviewee.get_top_k_similar_context(query_embedding, embeddings, data)
                prompt = interviewee.generate_prompt(tokenizer, context, query)
                output_text = interviewee.generate_answer(tokenizer, interviewee.load_llm_model(), prompt)
                clean_output_text = interviewee.clean_output_text(output_text)
                st.write(clean_output_text)

        else:
            generator_type = st.radio("Choose the type of the interviewer: ", ["Free", "Paid"])
            if generator_type == "Paid":
                api_key = st.text_input("Please enter the API key for the OpenAI platform", type="password")
                if api_key:
                    difficulty = st.slider("Select the difficulty of the interview:", 0, 10)
                    if difficulty:
                        interviewer = Interviewer(name, "paid", api_key, difficulty)
                        questions = interviewer.generate_questions()

                        # Get current question index
                        current_index = st.session_state.current_question_index
                        if current_index < len(questions):
                            question = questions[current_index]
                            st.write(f"{question}")

                            # Store the answer for this question in session state
                            answer_key = f"answer_{current_index}"
                            if answer_key not in st.session_state:
                                st.session_state[answer_key] = ""  # Initialize answer if not present

                            # Now call the text_area with the correct session state value
                            st.text_area("Your Answer", value=st.session_state[answer_key], key=answer_key)

                            if st.button("Submit Answer", key=f"submit_{current_index}"):
                                if len(st.session_state[answer_key].strip()) > 0:
                                    # Append the answer to session state
                                    st.session_state.answers.append(st.session_state[answer_key])

                                    # Score the answer
                                    score = interviewer.score_answer(question, st.session_state[answer_key])
                                    st.session_state.scores.append(int(score))
                                    st.write(f"The score for previous answer is {score} out of 10.")
                                    time.sleep(3)

                                    # Move to the next question
                                    st.session_state.current_question_index += 1
                                else:
                                    st.error("Please provide an answer before proceeding.")
                        else:
                            # Calculate the average score
                            if len(st.session_state.scores) > 0:
                                average_score = sum(st.session_state.scores) / len(st.session_state.scores)
                                st.write(f"Your interview score is {average_score:.2f} out of 10.")
                            st.success("The interview is over!")
                            st.write("Your answers:")
                            for i, ans in enumerate(st.session_state.answers, 1):
                                st.write(f"{i}. {ans}")

            else:  # Free interviewer mode
                interviewer = Interviewer(name, "free")
                questions = interviewer.generate_questions()

                # Get current question index
                current_index = st.session_state.current_question_index
                if current_index < len(questions):
                    question = questions[current_index]
                    st.write(f"{question}")

                    # Store the answer for this question in session state
                    answer_key = f"answer_{current_index}"
                    if answer_key not in st.session_state:
                        st.session_state[answer_key] = ""  # Initialize answer if not present

                    # Now call the text_area with the correct session state value
                    st.text_area("Your Answer", value=st.session_state[answer_key], key=answer_key)

                    if st.button("Submit Answer", key=f"submit_{current_index}"):
                        if len(st.session_state[answer_key].strip()) > 0:
                            st.session_state.answers.append(st.session_state[answer_key])
                            st.session_state.current_question_index += 1
                        else:
                            st.error("Please provide an answer before proceeding.")
                else:
                    st.success("The interview is over!")
                    st.write("Your answers:")
                    for i, ans in enumerate(st.session_state.answers, 1):
                        st.write(f"{i}. {ans}")