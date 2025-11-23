from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import DuckDuckGoSearchRun
from tools import search_tool

load_dotenv()

class reserchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

llm = ChatOpenAI(model="gpt-4o-mini")
parser = PydanticOutputParser(pydantic_object=reserchResponse)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            you are a research assistant that will help genrate a research paper.
            answer the user query and use neccessary tools.
            wrap the output in this format and provide no other text\n{format_instructions}
            """, 
        ),
       ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]    
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools, 
)

agent_executor = AgentExecutor(agent=agent, tools=[], verbose=True)
query = input("what can i help you research? ")
raw_response = agent_executor.invoke({"query": query,})
structured_response = parser.parse(raw_response["output"])
print(structured_response)