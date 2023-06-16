
# The following code is for the server of the calculator. This side contains the code that calculates the 
# answer to the inputted equation.

import socket
import math # Import math Module

_vServer = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # AF_INET corresponds to IPv4 SOCK_STREAM to TCP
_vServer.bind((socket.gethostname(), 10001)) # Assigns an IP address and a Port Number to a Socket Instance

_vServer.listen() # Listen for the Client
print("Listening for client...") 
_vClient, cl_addr = _vServer.accept() # Connect to the Client
print(f"Client connection established! : {cl_addr}") # Inform the User of the Client connection + Address
print(f"Socket is listening...") 

invalidequation = "Please enter a Valid Equation!"

while True:

    m = _vClient.recv(1024) # Maximum number of Bytes recieved from the Client
    m = m.decode("utf-8")
    
    m = m.replace("^", "**") # Replace the ^ input with **
    m = m.replace("cbrt", "**(1/3)") # Replace the cbrt input with a cube root expression
    m = m.replace("sqrt", "math.sqrt") # Replace the sqrt input with math.sqrt 
    m = m.replace("!", "math.factorial") # Replace the ! input with math.factorial 
    m = m.replace("e", "math.exp") # Replace the e input with math.exp
    m = m.replace("log2", "math.log2") # Replace the log2 input with math.log2
    m = m.replace("log10", "math.log10") # Replace the log10 input with math.log10

    try:
        _vClient.send(bytes(str(eval(m)), "utf-8")) # Evaluate the code sent as string and send back to the Client
        print("Valid Calculation Recieved!")
        continue

    except:
        _vClient.send(bytes(str(invalidequation), "utf-8"))
        print("Invalid Calculation Recieved!")
        continue 
    
    _vServer.listen()