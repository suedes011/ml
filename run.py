from urllib import request

print("MENU")
print("------------------------------------------")
print("1. SpamHam Sigmoid")
print("2. Naive Bayes")
print("3. Perceptron")
print("4. All in one")

ch = int(input("Enter your choice : "))
file = ""
if ch == 1:
    file = "spam"
elif ch == 2:
    file = "nb"
elif ch == 3:
    file = "pt"
elif ch == 4:
    file = "aio"
else:
    print("wrong option")
    
data = request.urlopen(f"https://raw.githubusercontent.com/suedes011/ml/main/{file}.py")
print("------------------------------------------\n")
print(data.read().decode('utf-8'))
