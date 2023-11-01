from LLM_Functions import llm_function

print("enter 'q' to exit")

while True:
    user_input = input("User: ")

    if user_input.lower() == "q":
        break
    response = llm_function(user_input)
    output_text = ("AI:" + response)
    print(output_text)
