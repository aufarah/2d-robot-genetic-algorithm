import OpenGL
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from glmodule import *
import sys
from colorlib import *
import numpy as np

def anglecond(angle):
    return 180 - angle

def forwardkin():
    global LoR,position
    if (LoR==0):
        position[1] = position[0] + stickLength*np.cos(np.deg2rad(degree/2))
        position[2] = position[0] + 2*stickLength*np.cos(np.deg2rad(degree/2))
    else:
        position[1] = position[2] - stickLength*np.cos(np.deg2rad(degree/2))
        position[0] = position[2] - 2*stickLength*np.cos(np.deg2rad(degree/2))
    
def showScreen():
        global degree, theta, initDeg
        global k, LoR, times,unidle
        
        if(unidle==True):
            error = posTarget - position[1]
            print(error)
            print(theta)
            thinker.feedind([error,theta])
            LoR = thinker.output[0][0]
            theta = thinker.output[0][1]
            
            
            if(theta>180):
                theta = 180
            elif(theta<0):
                theta = 0
            
            print(LoR)
            print(theta)
            print()
            
            unidle = False
      
        #print(LoR)
        #print(theta)
        
        #theta = 50
        #theta = anglecond(theta)
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) # Remove everything from screen (i.e. displays all white)
        glClearColor(blueSky[0],blueSky[1],blueSky[2],1)
        glLoadIdentity()
        iterate()
        rectang(viewWidth,150,greenGrass)
        glTranslate(0,150,0)
        
        if (LoR == 0):
            glTranslatef(position[0],0,0)
            glRotatef(-90,0,0,1)            #the robot started horizontally (x direction) at first
            glRotatef(degree/2,0,0,1)
            rectangX(20,stickLength,wooden1)
            glTranslatef(0,stickLength,0)
            
            glRotatef(-degree,0,0,1)
            rectangX(20,stickLength,wooden2)
        else:
            #find difference between old and new positon, invert it so the right foot still in its position
            shift = -2*stickLength*np.cos(np.deg2rad(degree/2)) + position[2]       
            glTranslate(shift,0,0)
            
            glRotatef(-90,0,0,1)            #the robot started horizontally (x direction) at first
            glRotatef(degree/2,0,0,1)
            rectangX(20,stickLength,wooden1)
            glTranslatef(0,stickLength,0)
            
            glRotatef(-degree,0,0,1)
            rectangX(20,stickLength,wooden2)
        
        forwardkin()
        #print(position)
        
        if(times>=0 and times<1):
            degree = initDeg + (anglecond(theta) - initDeg)*times/1            #move in 1 sec
        else:
            initDeg = anglecond(theta)
            times = 0
            unidle = True
            
        time.sleep(0.01)
        glutSwapBuffers()
        times+=0.01

class NeuralNetwork:
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def sigmoid_derivative(self, y):
        return y * (1 - y)

    def step(self,x):
        return 1.*(x>=0)
        
    def relu(self, x):
        return x*(x>=0)
        
    def drelu(self, y):
        return 1.*(y>=0)

    def __init__(self, insize, outsize):
        self.h2          = np.array([np.zeros(6)]).T                  #hidden layer 2, ada 6 neuron
        self.h1          = np.array([np.zeros(6)]).T                  #hidden layer 1, ada 6 neuron
		# 2|6|2
        #self.randominit(insize,outsize)
        self.smartinit()
        
        self.output     = np.zeros([1,outsize])

    def randominit(self, insize,outsize):
        self.weights3   = np.random.uniform(-1,1,[outsize, self.h2.shape[0]] )      #horizontal per input m, verticaled as y n
        self.bias3      = np.array([np.random.uniform(-1,1,outsize)]).T
        self.bias2      = np.array([np.random.uniform(-1,1,self.h2.shape[0])]).T
        self.weights2   = np.random.uniform(-1,1,[self.h2.shape[0], self.h1.shape[0]] ) 
        self.bias1      = np.array([np.random.uniform(-1,1,self.h1.shape[0])]).T
        self.weights1   = np.random.uniform(-1,1, [self.h1.shape[0], insize])
        
    def smartinit(self):
        self.weights1   = np.array([[-0.2548603 ,-0.96277833],[ 0.31122682, 0.44800976],[ 0.08702691, 0.08788676],[ 0.04811313, 0.0485885 ],[ 0.15608471, 0.76901224],[ 0.72494972,-0.96455403]])
        self.bias1      = np.array([[-0.30778828],[ 0.22143036],[-0.03848076],[ 0.59449392],[ 0.10197827],[-0.46506317]])
        self.weights2   = np.array([[-0.45441733, 0.52844625, 0.42381352, 0.4391193,  0.64433215,-0.65425405],[ 0.14540573, 0.00555961,-0.58135092,-0.47760229,-0.08920864,-0.1212385 ],[-0.69999701, 0.62537112,-0.81286171,-0.93078141, 0.149569, -0.0715295 ],[-0.24269141,-0.56720112, 0.29380841, 0.74311491,-0.29100074, 0.98315474],[-0.09327242,-0.50988857,-0.28903618, 0.76320526, 0.95330046, 0.0740337 ],[ 0.3933041,  0.16934115, 0.14487126, 0.39274503, 0.00954084,-0.3279348 ]])
        self.bias2      = np.array([[-0.01543831],[-0.00232167],[-0.14482197],[-0.0761846 ],[ 0.09080579],[ 0.02736207]])
        self.bias3      = np.array([[0.40956255],[0.0710337 ]])
        self.weights3   = np.array([[ 0.17165282,-0.49747152,-0.30173567,-0.56279934, 0.18025865,-0.18126395],[ 0.04739371,-0.04795577,-0.75267264, 0.70268893,-0.08585454, 0.18302092]])

    def feedind(self, input):
        #input must be a horizontal array with insize size: error, oldtheta
        sumh1 = np.matrix(self.weights1) * np.matrix(input).T
        outh1 = sumh1 + self.bias1
        outh1 = np.array(outh1)
        self.h1 = self.relu(outh1)
        
        sumh2 = np.matrix(self.weights2) * np.matrix(self.h1)
        outh2 = sumh2 + self.bias2
        outh2 = np.array(outh2)
        self.h2 = self.relu(outh2)

        sumir = np.matrix(self.weights3) * np.matrix(self.h2)
        out   = sumir + self.bias3
        out[0]  = self.step(out[0])          #for LoR
        out[1]  = self.relu(out[1])             #for degree
        #output then be horizontal-ed as an array with outsize size: LoR and newtheta    
        self.output = np.array(out).T
        self.output = np.array(self.output)
        
def keyboard(key,i,j):
    key = key.decode("utf-8")
    global posTarget
    if key=='a':
        posTarget += 200
    if key=='s':
        posTarget -= 200
    if key==chr(27):
        sys.exit()
        
stickLength = 100
LoR = 0
theta  = 0         #initial theta = 0
degree = 0         #start degree = 0
initLeft = 0
times = 0
unidle = True
initDeg = anglecond(theta)
#stick's head must be in x=posTarget
posTarget = 400
position = [initLeft, initLeft + stickLength*np.cos(np.deg2rad(initDeg/2)), initLeft + 2*stickLength*np.cos(np.deg2rad(initDeg/2))]
thinker = NeuralNetwork(2,2)
if __name__ == "__main__":
    init()
    glutDisplayFunc(showScreen)  # Tell OpenGL to call the showScreen method continuously
    glutKeyboardFunc(keyboard)
    glutIdleFunc(showScreen)     # Draw any graphics or shapes in the showScreen function at all times
    glutMainLoop()  # Keeps the window created above displaying/running in a loop
