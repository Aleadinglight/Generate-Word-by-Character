import numpy as np
import PWNet
import WordProcessing



def Generate_Word(model,c):
    
    new_word = [letter_to_index['+'], letter_to_index[c]]
    word=['+',c]
    next_letter_id = 0
    k=0
    while not new_word[-1]==28:
        k+=1
        next_letter_probs, s = model.feedforward(new_word)

        next_letter_id = np.argmax(next_letter_probs[-1])
        next_letter = index_to_letter[next_letter_id]
        
        new_word.append(next_letter_id)
        word.append(next_letter)
    return word

p = WordProcessing.Process()
X_train, Y_train = p.giveInput()
model = PWNet.RNNnp(29)
index_to_letter, letter_to_index = p.Dictionary()

c=raw_input("Enter a letter: ")
ans =  Generate_Word(model,c)
st=''
for i in ans[1:-1]:
    st=st+i;
print "\n",st

