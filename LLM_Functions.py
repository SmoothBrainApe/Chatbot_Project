from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryBufferMemory


model_path = "./model/zephyr-7b-beta.Q5_K_M.gguf"
template = ("""<|im_start|>system
           <|im_end|>
            {chat_history}
            <|im_start|>user
            {user_input}<|im_end|>
            <|im_start|>assistant""")
n_gpu_layers = 35
n_batch = 512
n_ctx = 16384
max_tokens = 1024
seed = 0
temperature = 1.31
repeat_penalty = 1.17
top_k = 49
top_p = 0.14
stop = [">user", "<|im_end|>"]
max_token_limit = 1024

# LLM variable setup with LlamaCPP
llm = LlamaCpp(
    model_path=model_path,
    max_tokens=max_tokens,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=n_ctx,
    seed=seed,
    temperature=temperature,
    repeat_penalty=repeat_penalty,
    top_p=top_p,
    top_k=top_k,
    stop=stop,
    verbose=False,
        )

# Memory setup with summary buffer
prompt = PromptTemplate(input_variables=["chat_history", "user_input"], template=template)
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=max_token_limit, memory_key="chat_history")
llm_chain = LLMChain(prompt=prompt, llm=llm, memory=memory, verbose=False)


def llm_function(message):  # basic llm chat function
    response = llm_chain.predict(user_input=message)
    output = response
    return output



