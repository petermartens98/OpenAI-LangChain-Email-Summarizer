import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain


def display_file_code(filename):
    with open(filename, "r") as file:
        code = file.read()
    with st.expander(filename, expanded=False):
        st.code(code, language='python')


def email_summarizer(email, subject):
    llm = OpenAI(model_name='gpt-3.5-turbo', temperature=1, max_tokens=256)
    templates = {
        'sender': "Determine the sender from the following subject: {subject}\n\n and email:\n\n{email}\n\nSender:",
        'role': "Determine the role of the sender from the following subject: {subject}\n\n and email:\n\n{email}\n\nRole:",
        'tone': "Provide the overall tone from the following subject: {subject}\n\n and email:\n\n{email}\n\nTone:",
        'summary': "Write a brief summary from the following subject: {subject}\n\n and email:\n\n{email}\n\nSummary:",
        'spam': "Determine if the following email is spam. I am a developer dealing with new clients, bussiness connections\
                and financial transactions, a lot of links are shared, is this email spam or not:\n\\n subject: {subject}\
                \n\nemail:\n{email}\n\nIs Spam?:",
    }
    outputs = {}
    for key, template in templates.items():
        prompt = PromptTemplate(input_variables=["email","subject"], template=template)
        chain = LLMChain(llm=llm, prompt=prompt)
        output = chain.run(email=email, subject=subject)
        outputs[key] = output

    with st.expander("Email Summary", expanded=True):
        for key, output in outputs.items():
            st.subheader(f"{key.capitalize()}:")
            st.write(output, end="\n\n")
        
def main():
    st.title("AI Email Summarization Tool")
    st.header("Powered by OpenAI, Langchain, Streamlit")
    deploy_tab, code_tab= st.tabs(["Deployment", "Code"])
    with deploy_tab:
        subject = st.text_input("Subject:")
        email =  st.text_area("Email:", height=300 )
        if st.button("Summarize Email"): email_summarizer(email, subject)
    with code_tab:
        st.header("Source Code")
        display_file_code("main.py")


if __name__ == "__main__":
    load_dotenv()
    main()
