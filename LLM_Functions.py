from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory


model = "Path/To/Model"  # GGUF model suggested. Format for mistral fine tunes preloaded.
model_type = "mistral"
template = ("""<|im_start|>system 
            <|im_end|>
            {chat_history}
            <|im_start|>user
            {user_input}<|im_end|>
            <|im_start|>assistant""")  # just add your prompt template on line 11 before <|im_end|>
gpu_layers = 35  # adjust the parameters according to your wish
threads = 8
context_length = 8192
max_new_tokens = 1024
temperature = 0.8
repetition_penalty = 1.1
top_k = 40
top_p = 0.95

llm = CTransformers(
        model=model,
        model_type=model_type,
        config={'max_new_tokens': max_new_tokens,
                'repetition_penalty': repetition_penalty,
                'top_k': top_k,
                'top_p': top_p,
                'temperature': temperature,
                'gpu_layers': gpu_layers,
                'threads': threads,
                'context_length': context_length,
                'stop': ["/s", "<|im_end|>"], })

prompt = PromptTemplate(input_variables=["chat_history", "user_input"], template=template)
memory = ConversationBufferMemory(memory_key="chat_history")
llm_chain = LLMChain(prompt=prompt, llm=llm, memory=memory, verbose=False)


def llm_function(message):
    response = llm_chain.predict(user_input=message)
    output = response
    return output
