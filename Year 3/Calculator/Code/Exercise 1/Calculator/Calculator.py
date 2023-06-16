# The following code serves as a calculator that performs the basic, and some more advanced, operations.
# It was coded with user simplicity in mind, so the software should be easy to use and understand.
# The client and the server work hand in hand to take in inputs and calculate the answer to the equation before 
# the output is displayed to the user. The section contains the code for the client.

from tkinter import * # Module for creating GUI
import socket
import sys # System module 
from datetime import datetime # Import datetime Module

_vC = socket.socket(socket.AF_INET, socket.SOCK_STREAM)     
_vC.connect((socket.gethostname(), 10001)) # Connect to the server

print("Welcome to the Calculator!")
print("This Calculator performs the basic Operations (+, -, / and *) as well as some other")
print("operations! There are steps on how to use the different operations below. Have Fun!\n")

howtouse = open("How_To_Use.txt", "r") # Open the Text File called 'How_To_Use'
for x in howtouse: # Loop through the lines of the File to read and print the whole File
    print(x)
 
expression = ""
invalidequation = "Please enter a Valid Equation!" # Define variables
 
def press(num): # This function updates the expression in the text box
    global expression
    expression = expression + str(num)
    equation.set(expression)
 
def equalpress(): # This function evaluates the expression inputted into the calculator
        global expression
        
        _vC.send(bytes(expression, "utf-8"))# Send the Equation to the Server
        m = _vC.recv(1024) # Maximum number of Bytes recieved
        total = m .decode("utf-8")
        
        if(total == invalidequation): # If an invalid equation was inputted, print 'invalidequation' then continue the loop
            print(total,"\n")
        else: 
            print(expression,"=",total,"\n")
            text_file = open("Answers.txt", "a") # Open the Text File 'Answers'
            text_file.write(f'{expression}={total}\n') # Save the Equation and Answer to the File
            text_file.close() # Close the Text File
            equation.set(total)
            expression = ""

def clear(): # This function clears the calculator log
    global expression
    expression = ""
    equation.set("")

def close(): # This function shuts down the system
    sys.exit()
 
if __name__ == "__main__": # Main loop

    now = datetime.now() # Get the System Date and Time
    dt = now.strftime("%d/%m/%Y %H:%M:%S") # Convert the Date and Time to UK formatting

    text_file = open("Answers.txt", "a") # Open or Create a Text File called 'Answers'
    text_file.write(f'{dt}\n') # Save the Date and Time to 'Answers'
    text_file.close()

    gui = Tk() # Create GUI
 
    gui.configure(background="dark red")
 
    gui.title("Calculator") # GUI name
 
    gui.geometry("263x203") # Parameters of GUI
 
    equation = StringVar()
 
    expression_field = Entry(gui, textvariable=equation)
 
    expression_field.grid(columnspan=4, ipadx=70)
 
    button1 = Button(gui, text=' 1 ', fg='black', bg='red', command=lambda: press(1), height=1, width=7) # Define buttons for GUI, this is button 1
    button1.grid(row=7, column=0) # the position of the button
 
    button2 = Button(gui, text=' 2 ', fg='black', bg='red', command=lambda: press(2), height=1, width=7) # This is button 2
    button2.grid(row=7, column=1)
 
    button3 = Button(gui, text=' 3 ', fg='black', bg='red', command=lambda: press(3), height=1, width=7) # This is button 3
    button3.grid(row=7, column=2)
 
    button4 = Button(gui, text=' 4 ', fg='black', bg='red', command=lambda: press(4), height=1, width=7) # This is button 4
    button4.grid(row=6, column=0)
 
    button5 = Button(gui, text=' 5 ', fg='black', bg='red', command=lambda: press(5), height=1, width=7) # This is button 5
    button5.grid(row=6, column=1)
 
    button6 = Button(gui, text=' 6 ', fg='black', bg='red', command=lambda: press(6), height=1, width=7) # This is button 6
    button6.grid(row=6, column=2)
 
    button7 = Button(gui, text=' 7 ', fg='black', bg='red', command=lambda: press(7), height=1, width=7) # This is button 7
    button7.grid(row=5, column=0)
 
    button8 = Button(gui, text=' 8 ', fg='black', bg='red', command=lambda: press(8), height=1, width=7) # This is button 8
    button8.grid(row=5, column=1)
 
    button9 = Button(gui, text=' 9 ', fg='black', bg='red', command=lambda: press(9), height=1, width=7) # This is button 9
    button9.grid(row=5, column=2)
 
    button0 = Button(gui, text=' 0 ', fg='black', bg='red', command=lambda: press(0), height=1, width=7) # This is button 0
    button0.grid(row=8, column=0)
 
    plus = Button(gui, text=' + ', fg='black', bg='red', command=lambda: press("+"), height=1, width=7) # This is button +
    plus.grid(row=4, column=0)
 
    minus = Button(gui, text=' - ', fg='black', bg='red', command=lambda: press("-"), height=1, width=7) # This is button -
    minus.grid(row=4, column=1)
 
    multiply = Button(gui, text=' * ', fg='black', bg='red', command=lambda: press("*"), height=1, width=7) # This is button *
    multiply.grid(row=4, column=2)
 
    divide = Button(gui, text=' / ', fg='black', bg='red', command=lambda: press("/"), height=1, width=7) # This is button /
    divide.grid(row=4, column=3)
 
    square = Button(gui, text=' ^ ', fg='black', bg='red', command=lambda: press("^"), height=1, width=7) # This is button ^
    square.grid(row=6, column=3)

    factorial = Button(gui, text=' ! ', fg='black', bg='red', command=lambda: press("!"), height=1, width=7) # This is button !
    factorial.grid(row=5, column=3)

    e = Button(gui, text=' e ', fg='black', bg='red', command=lambda: press("e"), height=1, width=7) # This is button e
    e.grid(row=7, column=3)

    log2 = Button(gui, text=' log2 ', fg='black', bg='red', command=lambda: press("log2"), height=1, width=7) # This is button log2
    log2.grid(row=3, column=2)

    log10 = Button(gui, text=' log10 ', fg='black', bg='red', command=lambda: press("log10"), height=1, width=7) # This is button log10
    log10.grid(row=3, column=3)

    openbracket = Button(gui, text=' ( ', fg='black', bg='red', command=lambda: press("("), height=1, width=7) # This is button (
    openbracket.grid(row=3, column=0)

    closebracket = Button(gui, text=' ) ', fg='black', bg='red', command=lambda: press(")"), height=1, width=7) # This is button )
    closebracket.grid(row=3, column=1)

    squareroot = Button(gui, text=' sqrt ', fg='black', bg='red', command=lambda: press("sqrt"), height=1, width=7) # This is button sqrt
    squareroot.grid(row=2, column=1)

    cuberoot = Button(gui, text=' cbrt ', fg='black', bg='red', command=lambda: press("cbrt"), height=1, width=7) # This is button cbrt
    cuberoot.grid(row=2, column=2)

    equal = Button(gui, text=' = ', fg='black', bg='red', command=equalpress, height=1, width=7) # This is button equal
    equal.grid(row=8, column=3)
 
    clear = Button(gui, text='Clear', fg='black', bg='red', command=clear, height=1, width=7) # This is button clear
    clear.grid(row=2, column='3')

    close = Button(gui, text='Close', fg='black', bg='red', command=close, height=1, width=7) # This is button close
    close.grid(row=2, column='0')
 
    Decimal= Button(gui, text='.', fg='black', bg='red', command=lambda: press('.'), height=1, width=7) # This is button .
    Decimal.grid(row=8, column=1)

    gui.mainloop() # Loop
