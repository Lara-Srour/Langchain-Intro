**I.	LangChain**  
**1.	Definition:** open-source framework for building applications based on large language models (LLMs).
**2.	Components**
*Chains:* the fundamental principle that holds various AI components in LangChain to provide context-aware responses. uses like connecting to different data sources, and generating unique content.
from langchain.chains import create_sql_query_chain
from langchain.chains import SQLDatabaseSequentialChain  # useful if the number of tables in database is large.
chain = create_sql_query_chain(llm, db)

*Prompt templates:* pre-built structures developers use to consistently and precisely format queries for AI models
from langchain_core.prompts import PromptTemplate

*Agents:* special chain that prompts the language model to decide the best sequence in response to a query. It creates complex LLM chain calls for answering user questions.
from langchain.agents import create_sql_agent

*Retrieval modules:* LangChain enables the architecting of RAG systems with numerous tools to transform, store, search, and retrieve information that refine language model responses.

**II.	LangChain with SQL**
LangChain comes with a number of built-in chains and agents that are compatible with any SQL dialect supported by SQLAlchemy, they enable use cases such as:
•	Generating queries that will be run based on natural language questions,
•	Creating chatbots that can answer questions based on database data,
•	Building custom dashboards based on insights a user wants to analyze,

![Alt text](images/sql_agent.png)



