import os
from apikey import apikey

import streamlit as st
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import initialize_agent, tool
from langchain.agents import load_tools
from docx import Document

os.environ['OPENAI_API_KEY'] = apikey

llm=OpenAI(temperature=0.1, model_name="gpt-4")

tools = load_tools(["serpapi"], llm=llm, serpapi_api_key = "478edbe21c14cf011cddcd9e3bd894f2885b77426a549434e6ea78bfee05b621")

prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools, 
    prefix=prefix, 
    suffix=suffix, 
    input_variables=["input", "chat_history", "agent_scratchpad"]
)
memory = ConversationBufferMemory(memory_key="chat_history")

llm_chain = LLMChain(llm=llm, prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)

st.title("Data Rebels AI Feedbacker  🤖")

code = st.text_input("Copia y Pega to codigo aquí!")

if st.button('Revisar Actividad!'):
    res = agent_chain.run("Eres un profesor experto en Python, un alumno te ha dado el siguiente codigo como su tarea {code} Debes determinar si el codigo es funcional y asignarle una calificacion entre el 1 y el 10. Considera que el alumno tendra 2.5 puntos por cumplir con cada uno de los siguientes criterios: Funciona el codigo, la sintaxis es correcta, el codigo es optimo, el codigo existe")
    st.header("Calificacion")
    st.write(res)
    retro = agent_chain.run("Eres un profesor experto en Python, un alumno te ha entregado el siguiente codigo como su tarea {code} Debes darle retroalimentacion al estudiante, dandole consejos, tips, recomendaciones y retroalimentacion para que el a partir de ahora pueda escribir mejor codigo, si consideras que la tarea que te ha entregado es excelente diselo")
    st.header("Retroalimentacion")
    st.write(retro)
    so = agent_chain.run("Eres un profesor experto en python y un alumno te ha entregado el siguiente codigo como su tarea {code} Describe con base e su codigo cuales son sus fortalezas al programar, sus areas de oportunidad y recomiendale una o dos bibliografias para mejorar sus areas de oportunidad")
    