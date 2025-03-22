import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMChain , LLMMathChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool , initialize_agent
from langchain.callbacks import StreamlitCallbackHandler


## Set UPI the streamlit app
st.set_page_config(page_title = "Text To MAth Problem Solver And Data Search Assistant",
page_icon = "🧮")
st.title("Text To Math Problem Solver Using Google Gemma 2")

groq_api_key = st.sidebar.text_input(label = "Groq API Key",type = "password")

if not groq_api_key :
    st.info("Please provide valid groq api key to continue")
    st.stop()

llm = ChatGroq(model = "Gemma2-9b-It", groq_api_key = groq_api_key)

# Initializing the tools

wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name = "Wikipedia",
    func = wikipedia_wrapper.run,
    description = "A tool for searching the internet to find various information on topics mentioned"
)

# Initialize the math tool
math_chain = LLMMathChain.from_llm(llm = llm)
calculator = Tool(
    name = "Calculator",
    func = math_chain.run,
    description = "A tool for answering maths related questions. Only mathematical expression need to be provided"
)

prompt = """
you are a agent tasked for solving mathematical questions. Logically arrive at the solution and 
provide a detailed explaination and display it point wise for the question below
Question : {question}
Answer : 
"""

prompt_template = PromptTemplate(
input_variables = ["question"],
template = prompt
)

# Combining all the tools in the chain
chain = LLMChain(llm = llm , prompt = prompt_template)

reasoning_tool = Tool(
    name = "Reasoning tool",
    func = chain.run,
    description  = "A for answering logic based and reasoning questions"
)

## Initialize the agents
assistant_agent = initialize_agent(
    tools = [wikipedia_tool,calculator,reasoning_tool],
    llm = llm,
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose = False,
    handel_parsing_errors = True
)

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi, I'm a math chatbot who can answer all your math related questions"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


#Lets start interaction
question = st.text_area("Enter Your Question :")

if st.button("Find My Answer"):
    if question:
        with st.spinner("Generate Response ."):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)

            st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)   # Responses appear dynamically instead of waiting for entire process to complete
            response = assistant_agent.run(st.session_state.messages , callbacks = [st_cb])  # Retrieves the response based on the stored conversation history

            st.session_state.messages.append({"role":"assistant","content":response})    # To maintain the chat history
            st.write("### Response :")
            st.success(response)
    else:
        st.warning("Please enter the question.")