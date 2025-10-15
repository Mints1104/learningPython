from Question import Question

questions_prompts = [
    "What color are apples?\n(a) Red/Green\n(b) Purple\n(c) Orange\n",
    "What color are bananas?\n(a) Teal\n(b) Magenta\n(c) Yellow\n",
    "What color are strawberries?\n(a) Yellow\n(b) Red\n(c) Blue\n"
]

questions = [
    Question(questions_prompts[0], "a"),
    Question(questions_prompts[1],"c"),
    Question(questions_prompts[2],"b")
]

def run_quiz(questions):
    score = 0
    for question in questions:
        print(f"Question: {question.prompt}\n Please type your answer below")
        userInput = input()
        if userInput == question.answer:
            score +=1 
            print(f"Correct! \nNew Score: {score}")
        else:
            print("Incorrect answer :(")
    print(f"Quiz finished \nFinal Score: {score}")
    
run_quiz(questions)