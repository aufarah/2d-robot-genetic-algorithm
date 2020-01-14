import numpy as np
import copy
 
numChild = 10
numGeneration = 40
initialpop = 20
        
def anglecond(angle):
    return 180 - angle

def forwardkin():
    global LoR,position
    if (LoR==0):
        position[1] = position[0] + stickLength*np.cos(np.deg2rad(anglecond(theta)/2))
        position[2] = position[0] + 2*stickLength*np.cos(np.deg2rad(anglecond(theta)/2))
    else:
        position[1] = position[2] - stickLength*np.cos(np.deg2rad(anglecond(theta)/2))
        position[0] = position[2] - 2*stickLength*np.cos(np.deg2rad(anglecond(theta)/2))
        
def applicate(individu):
    #applicate the THINK (network)
    global error, position, LoR, theta, initLeft, initDeg
    initLeft = 0
    initDeg = anglecond(0)
    position = [initLeft, initLeft + stickLength*np.cos(np.deg2rad(initDeg/2)), initLeft + 2*stickLength*np.cos(np.deg2rad(initDeg/2))]
    position = [0, 0, 0]
    for steps in range(10):
        #error is differences between target and now head position
        error = posTarget - position[1]
        
        #update
        individu.feedind([error,theta])
        LoR = individu.output[0][0]
        theta = individu.output[0][1]
        print(error)
        if(theta>180):
            theta = 180
        elif(theta<0):
            theta = 0
        
        #get the new position
        forwardkin() 
        errorl = posTarget - position[1]
        cost = errorl**2
        cost = cost**0.5
        print(LoR)
        print(theta)
        print(cost, ", ", position[1])
    return [cost,position[1]]
        
def sort(input,output2, output):
    for i in range(len(output)):
        lowIndex = i
        for j in range(i+1,len(output)):
            if output[j] < output[lowIndex]:
                lowIndex = j

        output[i],output[lowIndex] = output[lowIndex],output[i]
        output2[i],output2[lowIndex] = output2[lowIndex],output[i]
        input[i],input[lowIndex] = input[lowIndex],input[i]
    return [input,output2,output]
        
def havesex(population):
    for i in range(numChild):
        #choosing parent
        p1 = np.random.randint(len(population))
        p2 = np.random.randint(len(population))

        #switching parameters (gen) between parent
        child = copy.deepcopy(population[p1])

        #swapping 5 element randomly
        """weights1 size 6x2"""
        limxW1 = child.weights1.shape[0]
        limyW1 = child.weights1.shape[1]
        chosen= np.random.choice(limyW1,2,replace=False)
        y = 0
        z = 1
        constnar = np.random.uniform(-y,z)
        child.weights1[np.random.randint(limxW1)][chosen]  = constnar * population[p2].weights1[np.random.randint(limxW1)][chosen]
        
        """bias1 size 6x1"""
        limxW1 = child.bias1.shape[0]
        limyW1 = child.bias1.shape[1]
        chosen= np.random.choice(limxW1,3,replace=False)
        constnar = np.random.uniform(-z,z)
        child.bias1[chosen]      = constnar * np.random.uniform(0,2) * population[p2].bias1[chosen]
        
        """weights2 size 6x6"""
        limxW1 = child.weights2.shape[0]
        limyW1 = child.weights2.shape[1]
        chosen= np.random.choice(limyW1,3,replace=False)
        constnar = np.random.uniform(-y,z)
        child.weights2[np.random.randint(limxW1)][chosen]  = constnar * population[p2].weights2[np.random.randint(limxW1)][chosen]
        
        """bias2 size 6x1"""
        limxW1 = child.bias2.shape[0]
        limyW1 = child.bias2.shape[1]
        chosen= np.random.choice(limxW1,3,replace=False)
        constnar = np.random.uniform(-y,z)
        child.bias2[chosen]      = constnar  * population[p2].bias2[chosen]
        
        """weights3 size 2x6"""
        limxW1 = child.weights3.shape[0]
        limyW1 = child.weights3.shape[1]
        chosen= np.random.choice(limyW1,3,replace=False)
        constnar = np.random.uniform(-y,z)
        child.weights3[np.random.randint(limxW1)][chosen]   = constnar  * population[p2].weights3[np.random.randint(limxW1)][chosen]
        
        """bias3 size 2x1"""
        limxW1 = child.bias3.shape[0]
        limyW1 = child.bias3.shape[1]
        chosen= np.random.choice(limxW1,1,replace=False)
        constnar = np.random.uniform(-y,z)
        child.bias3[chosen]      = constnar * population[p2].bias3[chosen]
        
        population  = np.append(population,child)
    return population
    
def geneticStart(limitError):
    global population,newpop,fitness,finalpos
    for c in range(numGeneration):
        #mating
        population = havesex(population)

        #find fitness of populations
        fitness = np.array([])
        finalpos= np.array([])

        for j in range(len(population)):
            #print("This is no. ", j)
            [fitnessIndiv, finalposIndiv] = applicate(population[j])
            fitness = np.append(fitness,fitnessIndiv)
            finalpos= np.append(finalpos,finalposIndiv)

        #sorting
        [population,finalpos,fitness] = sort(population,finalpos,fitness)

        #kill the unfits
        initial = len(population)
        for i in range(numChild):
            index = i + 1
            population = np.delete(population,initial-index)
            fitness   = np.delete(fitness,initial-index)
            finalpos   = np.delete(finalpos,initial-index)
        #print(fitness[0],", ",finalpos[0])
        if(fitness[0]<limitError):
            newpop = np.append(newpop,population[0])
        if(fitness[0]<1):
            break
            
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
        
if __name__ == "__main__":
    stickLength = 100
    theta  = 0         #initial theta = 0
    degree = 0         #start degree = 0
    
    times = 0
    initDeg = anglecond(0)
    #stick's head must be in x=posTarget
    posTarget = 600
    initLeft = 0
    position = [initLeft, initLeft + stickLength*np.cos(np.deg2rad(initDeg/2)), initLeft + 2*stickLength*np.cos(np.deg2rad(initDeg/2))]
    
    ceq = NeuralNetwork(2,2)
    #making initial population
    """
    population = np.array([])
    newpop = np.array([])
    for i in range(initialpop):
        population = np.append(population, NeuralNetwork(2,2))
    
    geneticStart(10)
    #population = newpop
    #geneticStart(1)
    #population = newpop
    print("weights1: ")
    print(population[0].weights1)
    print("bias1: ")
    print(population[0].bias1)
    print("weights2: ")
    print(population[0].weights2)
    print("bias2: ")
    print(population[0].bias2)
    print("weights3: ")
    print(population[0].weights3)
    print("bias3: ")
    print(population[0].bias3)
    
    
    print("weights1: ")
    print(ceq.weights1)
    print("bias1: ")
    print(ceq.bias1)
    print("weights2: ")
    print(ceq.weights2)
    print("bias2: ")
    print(ceq.bias2)
    print("weights3: ")
    print(ceq.weights3)
    print("bias3: ")
    print(ceq.bias3)
    """
    applicate(ceq)
