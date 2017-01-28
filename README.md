# Generate-Word-by-Character

Recurrent Neural Network (RNN) is a type Neural Network which is mostly use for sequences predicting.

In this project I use a RNN to predict the next character of a word. Input will be some letters and the output will be the full word according to the previous letters.

I train this using Python Numpy. The input was processed by NLTK library in Python. The hidden unit size is 200. Output is a 29x1 vector determind the letter with highest probability to show up next: +abcd...xyz'- 

'+' and '-' was use as a start constant and an end constant. 

I also put a single quote in the set in case there is something like: " I'm, You're ". However, that was not really useful since NLTK somehow classified ' as a word - which I deleted since I don't want it. But since I have to be careful with ' appear from nowhere, I let ' inside the vocab to advoid error. I will check on how to make it better later.

This was train using Backpropagation Through Time. I used softmax function as the output layer.
