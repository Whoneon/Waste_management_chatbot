from dotenv import load_dotenv, find_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain_community.document_loaders import OnlinePDFLoader

import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler

# Streaming Handler
class StreamHandler(BaseCallbackHandler):
    def __init__(
        self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""
    ):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


load_dotenv(find_dotenv())


# Load and prepare documents
web_loader = WebBaseLoader(
    [
     "https://eur-lex.europa.eu/legal-content/IT/TXT/?uri=CELEX:02008L0098-20150731&from=ET" #DIRETTIVA 2008/98/CE in formato testuale
    ]
)

documents = [
	"https://www.srrcataniametropolitana.it/wp-content/uploads/2022/08/Definitivo_GEMA_Calendario_UtenzeDomestiche.pdf", #Calendario raccolta differenziata Città di Catania
	"https://www.srrcataniametropolitana.it/wp-content/uploads/2017/04/REGOLAMENTO-PER-LA-RACCOLTA-DIFFERENZIATA.pdf", #Normativa locale per la raccolta differenziata
	"https://dgdighe.mit.gov.it:5001/$DatiCmsUtente/normativa/leggi/DLgs%2003.04.2006%20n.152%20-%2010.09.2021.pdf", #DLgs 03.04.2006 n.152
	"https://www.albonazionalegestoriambientali.it/Download/it/NormativaNazionale/013-DLGSa152_03.04.2006_AllDparteIV_AGG.pdf", #Codici EER per la classificazione dei rifiuti
	"https://www.spp.cnr.it/images/FAQ-rifiuti.pdf", #FAQ sui rifiuti
	"https://eur-lex.europa.eu/legal-content/IT/TXT/PDF/?uri=CELEX:02008L0098-20150731&from=ET", #DIRETTIVA 2008/98/CE
	]
	
for file in documents:
	temp_pdf_loader = OnlinePDFLoader(file)
	pdf_docs = temp_pdf_loader.load()

web_docs = web_loader.load()

docs = pdf_docs + web_docs

documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=400
).split_documents(docs)
vector = Chroma.from_documents(documents, OpenAIEmbeddings())
retriever = vector.as_retriever(search_type="mmr", search_kwargs={"k": 6})

# Create tools
retriever_tool = create_retriever_tool(
    retriever,
    "Waste_management_helper",
    "Help users find information about waste management regulations",
)
# Search tool
search = TavilySearchResults(max_results=3)
tools = [search, retriever_tool]

# Initialize the model
model = ChatOpenAI(model="gpt-4o", streaming=True)

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer always in the language of the question",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Create the agent
agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Streamlit app
st.set_page_config(page_title="Waste Management Assistant", page_icon="♻️")
st.header("Your waste management assistant 🗑️")
st.write(
    """Ciao! Sono un assistente virtuale creato da Whoneon.
Ti aiuterò con qualsiasi domanda riguardo il mondo dei rifiuti, delle normative vigenti, e della raccolta dei rifiuti a Catania!
Chiedimi tutto ciò che desideri!"""
)


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with st.sidebar:
    st.header("Domande Frequenti:")
    st.markdown("- Quali sono le normative vigenti in merito allo smaltimento dei rifiuti?")
    st.markdown("- Come posso disporre correttamente dei rifiuti elettronici ingombranti?")
    st.markdown("- Quali sono le sanzioni per chi non rispetta tali leggi?")
    st.markdown("- Abito nel quartiere Borgo di Catania, in che giorno posso esporre il sacchetto dell'umido?")
    st.markdown("- Cosa si intende per indifferenziata a Catania? Posso buttarci anche le pile?")

    st.divider()
    
    st.markdown("Chiedimi tutto sulla raccolta dei rifiuti! Posso aiutarti con le normative:")
    st.markdown("DLgs 03.04.2006 n.152")
    st.markdown("Direttiva 2008/98/CE")
    st.markdown("Direttive comunali città di Catania")
    st.markdown("Tabella dei codici EER/CER per la catalogazione dei rifiuti")

if prompt := st.chat_input("Chiedimi qualcosa!", key="first_question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        
        stream_handler = StreamHandler(st.empty())
        # Execute the agent with chat history
        result = agent_executor(
            {
                "input": prompt,
                "chat_history": [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
            },
            callbacks=[stream_handler],
        )
        response = result.get("output")

    st.session_state.messages.append({"role": "assistant", "content": response})
    # st.chat_message("assistant").markdown(response)
