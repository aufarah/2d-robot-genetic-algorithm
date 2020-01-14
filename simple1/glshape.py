import OpenGL
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from glshape import *

def square(size,color):
    # We have to declare the points in this sequence: bottom left, bottom right, top right, top left
    glColor3f(color[0], color[1], color[2])
    glBegin(GL_QUADS) # Begin the sketch
    glVertex2f(0, 0) # Coordinates for the bottom left point
    glVertex2f(size, 0) # Coordinates for the bottom right point
    glVertex2f(size, size) # Coordinates for the top right point
    glVertex2f(0, size) # Coordinates for the top left point
    glEnd() # Mark the end of drawing
    
def rectang(width,height,color):
    glColor3f(color[0], color[1], color[2])
    glBegin(GL_QUADS)
    glVertex2f(0, 0) # Coordinates for the bottom left point
    glVertex2f(width, 0) # Coordinates for the bottom right point
    glVertex2f(width, height) # Coordinates for the top right point
    glVertex2f(0, height) # Coordinates for the top left point
    glEnd() # Mark the end of drawing
    
def rectangX(width,height,color):
    glColor3f(color[0], color[1], color[2])
    glBegin(GL_QUADS)
    glVertex2f(-width/2, 0) # Coordinates for the bottom left point
    glVertex2f(width/2, 0) # Coordinates for the bottom right point
    glVertex2f(width/2, height) # Coordinates for the top right point
    glVertex2f(-width/2, height) # Coordinates for the top left point
    glEnd() # Mark the end of drawing