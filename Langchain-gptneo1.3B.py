import torch
import tensorflow as tf
import transformers
from transformers import pipeline
from transformers import AutoTokenizer, GPTNeoForCausalLM
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
import sqlite3
import sqlalchemy as sa
import os


#Try Connection to Database
# Verify file existence and permissions
db_file = 'C:/Users/Admin/Desktop/Internship/chinook.db'
if not os.path.exists(db_file):
    print(f"Database file not found: {db_file}")
else:
    # Create the SQLDatabase object
    db = SQLDatabase.from_uri(f'sqlite:///{db_file}')
    print(db.dialect)
    print("Tables in the database:", db.get_usable_table_names())

#Load the pretrained Model gpt-neo-1.3B
model_name = "EleutherAI/gpt-neo-1.3B"
model = GPTNeoForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True)

# Ensure tokenizer is aligned with the model
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

#create the pipeline
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=500,
    torch_dtype=torch.float32,
    device=-1  # Or specify the device (GPU or CPU)
)

# Trying the model on a simple demo
llm = HuggingFacePipeline(pipeline=pipeline)
#prompt = PromptTemplate(
#    input_variables=["content"],
#    template = "Give an advice for an {content}."
#)

#create the chain
#chain = prompt | llm
#print(chain.invoke('Engineer')) 

#Integerate langchain with the Database and the LLM Model
generate_query = create_sql_query_chain(llm, db)
query = generate_query.invoke({"question": "How many employees are there"})
print(query)

