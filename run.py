from urllib import request

print("MENU")
print("------------------------------------------")
print("1. SpamHam Sigmoid")
print("2. Naive Bayes")

ch = int(input("Enter your choice : "))
file = ""
if ch == 1:
    file = "spam"
elif ch == 2:
    file = "nb"
else:
    print("wrong option")
    
data = request.urlopen(f"https://raw.githubusercontent.com/suedes011/ml/main/{file}.py")
print(data.read().decode('utf-8'))
