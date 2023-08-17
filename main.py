from dotenv import load_dotenv
from langchain.agents import AgentType, create_csv_agent, initialize_agent
from langchain.tools import Tool
from langchain.agents.agent_toolkits import create_python_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import PythonREPLTool

load_dotenv()


def main():
    python_agent_executor = create_python_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        tool=PythonREPLTool(),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    # python_agent_executor.run(
    #     "generate and save in current working directory 5 QRcodes that point to https://arxiv.org/abs/1706.03762, you have qrcode package installed already"
    # )

    csv_agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        path="english_movies.csv",
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    # csv_agent("Which movie is most popular in the file english_movies.csv")

    grand_agent = initialize_agent(tools=[
        Tool(
            name="PythonAgent",
            func=python_agent_executor.run,
            description="""useful when you need to transform natural language and write from it python and execute the python code
            returning the results of the code execution, 
            DO NOT SEND PYTHON CODE TO THIS TOOL
            """
        ),
        Tool(
            name="CSVAgent",
            func=csv_agent.run,
            description="""useful when you need to answer question over english_movies.csv file, 
            takes an input the entire question and returns the answer after running pandas calculations.
            """
        )
    ],
        llm=ChatOpenAI(temperature=0, model='gpt-3.5-turbo'), agent_type=AgentType.OPENAI_FUNCTIONS, verbose=True)

    grand_agent.run("Which movie is most popular in the file english_movies.csv")


if __name__ == "__main__":
    main()
