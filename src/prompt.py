from langchain_core.prompts import ChatPromptTemplate

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following piece of context to answer the question. "
    "Use three sentences maximum to answer the question and keep the answer concise."
    "\n\n"
    "{context}"
)

# Create the prompt template object
# We define "qa_prompt" which we will import in app.py
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)