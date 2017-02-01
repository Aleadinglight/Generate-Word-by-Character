from Tkinter import*
import Word_Predicting

class program():
    def __init__(self):
        self.root = Tk()
        self.root.bind("<Escape>",self.close)
        
        
        
        self.frameT=Frame(self.root)
        self.l1=Label(self.frameT,text="Input some letter: ").grid(row=0, column=0)      
        self.w=Entry(self.frameT, width=20)
        self.w.bind("<Return>",self.run)
        self.w.grid(row=0, column=1)
        self.frameT.pack()
        
        self.frameA=Frame(self.root)
        self.l2=Label(self.frameA, text="")
        self.l2.grid(row=0, column=0)
        self.frameA.pack()
        
        self.frameb=Frame(self.root)
        self.b = Button(self.frameb,text="OK", command=self.run)
        self.b.bind("<Return>", self.run)
        self.b.grid(row=0, column=0)
        self.c = Button(self.frameb,text="Exit", command=self.close).grid(row=0, column=1)
        self.frameb.pack()
        
        self.root.mainloop()
    
    def close(self, event=None):
        self.root.destroy()
    
    def run(self, event=None):
        st = Word_Predicting.Quest(self.w.get())
        self.l2.config(text=st)
        self.frameA.pack()
    
p=program()
