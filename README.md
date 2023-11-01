# Chatbot_Project

just a basic chatbot inference in the commandline. This is a side project that I just started to learn more about LLMs.

```
git clone https://github.com/SmoothBrainApe/Chatbot_Project
python -m venv venv
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
pip install -r requirements.txt
python chat.py
```

thats all. will add more features when I have time.
Feel free to use my code to edit. I hope you can fork so I can learn from your edits as well.

# UPDATE: Added RAG function but I am not sure how to incorporate it with the normal chat function. It is still currently a seperate function but it works for now. I want it to be seamless with the regular chat and document retrieval.
