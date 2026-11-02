from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model = ChatOpenAI()

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser
config={

    'tags':['llm app', 'report generation', 'summarization'],
    'metadata':{'model1':'gpt-40-mini', 'model1_temp':0.7, 'parser':'stroutputparser'}
}


#to set apne pasand ke tags and metadata in our trace - used for classifiying/extra info for understanding on langsmith
result = chain.invoke({'topic': 'Unemployment in India'},config=config)

print(result)
