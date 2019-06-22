import tkinter

def Disable_Buttons(root,Buttons):
	for b in Buttons:
		b.config(state='disabled')
	root.update()

def Enable_Buttons(root,Buttons):
	for b in Buttons:
		b.config(state='normal')
	root.update()

def Quit_Program(root):
	root.destroy()
	quit()

def Update_Label(root,Label,Label_Text):
	Label.config(text=Label_Text)
	root.update()

def Message(Metadata):
	tkinter.tkMessageBox.showinfo(Metadata["Title"],Metadata["Message"])
