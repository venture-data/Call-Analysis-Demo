import streamlit as st
import streamlit_scrollable_textbox as stx
import assemblyai as aai
import openai
import os
import json

ENVIRONMENT = "Production" # Local or Production

# Initialization
if 'transcript' not in st.session_state:
    st.session_state['transcript'] = ''
if 'analysis' not in st.session_state:
    st.session_state['analysis'] = ''
if 'analyzed' not in st.session_state:
    st.session_state['analyzed'] = 'false'


if ENVIRONMENT == 'Production':
    assembly_ai = st.secrets['ASSEMBLY_AI_KEY']
    open_ai = st.secrets['OPEN_AI_KEY']
    
elif ENVIRONMENT == 'Local':
    from dotenv import load_dotenv
    load_dotenv()
    assembly_ai = os.getenv('ASSEMBLY_AI_KEY')
    open_ai = os.getenv('OPEN_AI_KEY')


aai.settings.api_key = assembly_ai
client = openai.Client(api_key=open_ai)


def get_openai_response(question, transcript, model="gpt-3.5-turbo-0125"):
    response = client.chat.completions.create(
        model= model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that only answers from call transcript."},
            {"role": "user", "content": transcript},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content


def get_openAI_response(prompt, model="gpt-3.5-turbo-0125", temperature = 0):
  message = [{'role': 'user', 'content': prompt}]
  response = client.chat.completions.create(
      model=model,
      messages=message,
      temperature=temperature
  )
  return response.choices[0].message.content


# Streamlit app
def main():


    st.sidebar.image("venturedata-removebg-preview.png", width=284, channels="RGB")
    # Use HTML and CSS to center the title in the sidebar
    st.sidebar.markdown(
        "<h1 style='text-align: center;'>GenAI - Calls Analysis</h1>", 
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("Upload an audio file (WAV or MP3)", type=["wav", "mp3"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav' if uploaded_file.type == 'audio/wav' else 'audio/mp3')
        st.sidebar.success("File Uploaded")

        if st.button("Analyse"):

            st.sidebar.success("Analysis Started")
            transcript = aai.Transcriber().transcribe(uploaded_file)
            st.session_state.transcript = transcript.text
            st.session_state.analyzed = 'true'

            if transcript.text:

                prompt = f"""
                Given the transcript of a customer service call between a representative and a caller regarding plumbing services, determine whether the call outcome is Booked, Excused, or Unbooked.

                Please categorize the call based on the actions and outcomes described in the transcript:
                Definitions:
                - 'Booked': An appointment or job was successfully booked or an agent was sent to the customer.
                - 'Unbooked': An appointment or job was finalized or not booked nor an agent was sent to the customer. The callers may need further follow-up or consideration before committing to scheduling a service call or receiving a quote.
                - 'Not a Lead': The call outcome is Not a Lead because the caller primarily involve existing customers seeking assistance or information related to services provided by Cardinal Plumbing, Heating, and Air, resolving issues, or making personal inquiries, rather than representing potential leads for new business, customer confirming existing appintmemt, or providing information about service procedures without the intent to initiate a new service or expand upon an existing one.
                - 'Excused': The issue was resolved during the call, no appointment was necessary, and no agent was sent to the customer.

                Provide your reasoning for your decision in one sentence.

                Transcript:
                [The transcript of the call is provided within triple backticks.]

                ```{st.session_state.transcript}```

                Steps to follow:
                1. Analyze the transcript and determine the call category. But dont Give output yet.
                2. Provide "Booked", "Excused", "Not a Lead" or "Unbooked", without any additional text.
                3. Only provide output in json where the first line provide "Class" and the value is "Booked", "Excused", "Not a Lead" or "Unbooked". 
                Second line provide "Explanation" and the value is the reasoning for the decision. Third provide "Summary" of the call in 5 lines. 
                Lastly, provide "Entities" including 1. Customer Name, 2. address, 3. Services Requested, 4. Reason of call. 

                """
                analysis = get_openAI_response(prompt)
                st.session_state.analysis = analysis
    

if __name__ == "__main__":
    main()

    if "analysis" in st.session_state.keys():
        if len(st.session_state.analysis) > 0:
            data = json.loads(st.session_state.analysis)
            # Access elements separately
            class_value = data["Class"]
            explanation_value = data["Explanation"]
            summary_value = data["Summary"]
            entities = data['Entities']
            
            st.header("Analysis")

            st.subheader("Call Summary")
            st.markdown(summary_value)

            st.subheader("Class")
            st.markdown(class_value)

            st.subheader("Reasoning")
            st.markdown(explanation_value)

            st.subheader("Entities")
            for i, (k, v) in enumerate(entities.items()):
                with st.container():
                    col1, col2 = st.columns([2, 4])
                    with col1:
                        st.markdown(f"**{i+1}. {k}**")
                    with col2:
                        st.markdown(v)
            
            
            if class_value == "Booked":
                st.success("Trigger Sent.")
            else:
                st.error("Trigger Sent.")
            
            st.sidebar.success("Analysis Completed")


    if "transcript" in st.session_state.keys():
        if len(st.session_state.transcript) > 0:
            # Display the transcription
            st.subheader("Transcription:")
            # Use a container to hold the transcription
            stx.scrollableTextbox(st.session_state.transcript, height = 300)
            # st.text_area("", transcript.text, height=300, disabled=False)

            if st.session_state.analyzed == 'true':
                # Chat interface
                st.header("Ask Questions")
                user_question = st.text_input("Enter your question related to the call:")
                if st.button("Submit Question"):
                    if user_question:
                        
                        response = get_openai_response(user_question, st.session_state.transcript)
                        st.text_area("Response", value=response, height=150, disabled=True)
                    else:
                        st.warning("Please enter a question.")
