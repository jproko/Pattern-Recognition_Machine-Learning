# Package for scientific computing with Python
# https://numpy.org/
import numpy as np

# Matplotlib is a library for creating static, animated, and interactive visualizations in Python.
# https://matplotlib.org/
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

#=======================================================================================================================
def plotPerceptronResults(theta, x_i_x, x_i_y, x_i_z):
    
    #============================= PLOT XY ====================================
    # 1. Add subplot (rows, columns, index)
    ax = fig.add_subplot(121)

    #ax.set_xlim([100,210])
    # height
    #ax.set_ylim([45,110])

    # generate x axis
        #positive
    ax.arrow(0, 0, 210, 0.0, color="black")
        #negative
    ax.arrow(0, 0, -5, 0.0, color="black")

    # generate y axis
        #positive
    ax.arrow(0, 0, 0.0, 110, color="black")
        #negative
    ax.arrow(0, 0, 0.0, -5, color="black")

    # 2. Plot all the observation data points
    ax.plot(x_no, y_no, 'o', color='red')
    ax.plot(x_yes, y_yes,'o', color='green')
    ax.plot(x_i_x, x_i_y, 'v', color='black')

    ax.set_xlabel("x:Height (cm)")
    ax.set_ylabel("y:Weight (kg)")
    #==========================================================================
    #============== GENERATE 2 POINTS FOR THE CLASSIFIER LINE =================
    # equation:
    # x2 = -( x1*theta1 + 1*theta0 ) / theta2
    # y = -( x1*theta[0] + 1*theta[2] ) / theta[1]
    # for x= [120, 210]
    x = [-12,210]
    y = [0,0]
    y[0] = -( x[0]*theta[0] + 1*theta[2] ) / theta[1]
    y[1] = -( x[1]*theta[0] + 1*theta[2] ) / theta[1]
    ax.plot(x,y)
    #==========================================================================
    #============================= PLOT 3D ====================================
    # add subplot (3d projection of rows, columns, index)
    ax = fig.add_subplot(122, projection='3d')
    #ax.set_xlim([100,210])
    #ax.set_ylim([45,110])
    ax.set_zlim([-0.5,1.2])
    # generate x axis
        #positive
    ax.quiver(0.0, 0.0, 0.0, 220.0, 0.0, 0.0, arrow_length_ratio=0.001, alpha=0.5, color="black")
        #negative
    ax.quiver(0.0, 0.0, 0.0, -220.0, 0.0, 0.0, arrow_length_ratio=0.001, alpha=0.5, color="black")
    # generate x plane
    #x_x=[-250,250, 250, -250]
    #x_y=[0, 0, 0, 0]
    #x_z=[-5.0,-5.0, 5.0, 5.0]
    #vertices = [list(zip(x_x,x_y,x_z))]
    #poly = Poly3DCollection(vertices, alpha=0.1, color="black")
    #ax.add_collection3d(poly)

    #---------------------
    # generate y axis
        #positive
    ax.quiver(0.0, 0.0, 0, 0.0, 120.0, 0.0, arrow_length_ratio=0.001, alpha=0.5, color="black")
        #negative
    ax.quiver(0.0, 0.0, 0, 0.0, -120.0, 0.0, arrow_length_ratio=0.001, alpha=0.5, color="black")
    # generate y plane
    y_x=[-250,250, 250, -250]
    y_y=[-130, -130, 130, 130]
    y_z=[0,0, 0, 0]
    vertices = [list(zip(y_x,y_y,y_z))]
    poly = Poly3DCollection(vertices, alpha=0.1, color="black")
    ax.add_collection3d(poly)

    #---------------------
    # generate z axis
        #positive
    ax.quiver(0.0, 0.0, 0, 0.0, 0.0, 5.0, arrow_length_ratio=0.1, alpha = 0.5, color="black")
        #negative
    ax.quiver(0.0, 0.0, 0, 0.0, 0.0, -5.0, arrow_length_ratio=0.1, alpha = 0.5, color="black")
    # generate z plane
    #z_x=[0,0, 0, 0]
    #z_y=[-130, 130, 130, -130]
    #z_z=[-5.0,-5, 5.0, 5.0]
    #vertices = [list(zip(z_x,z_y,z_z))]
    #poly = Poly3DCollection(vertices, alpha=0.1, color="black")
    #ax.add_collection3d(poly)

    ax.scatter(x_no,y_no,z_no,'o', color='red')
    ax.scatter(x_yes,y_yes,z_yes,'o', color='green')
    ax.scatter(x_i_x, x_i_y, x_i_z,'v', color='black')

    ax.set_xlabel("x:Height (cm)")
    ax.set_ylabel("y:Weight (kg)")
    ax.set_zlabel("z:Constant 1")

    ax.quiver(0.0, 0.0, 0.0, theta[0], theta[1], theta[2], arrow_length_ratio=0.001, color="blue")
    perp = [190, 90, 0]
    perp[2] = -(perp[0]*theta[0]+perp[1]*theta[1])/theta[2]
    ax.quiver(0.0, 0.0, 0.0, theta[0], theta[1], theta[2], arrow_length_ratio=0.001, color="blue")
    ax.quiver(0.0, 0.0, 0.0, perp[0], perp[1], perp[2], arrow_length_ratio=0.0001, color="red")
    ax.view_init(15,45)
    #==========================================================================

    plt.show()
#=======================================================================================================================

# function that receives two vectors that can 
# be multiplied and returns their dot product
def dotProduct(theta_vector, data_point_x, data_point_y, data_point_z):
    result = 0
    result += theta_vector[0]*data_point_x
    result += theta_vector[1]*data_point_y
    result += theta_vector[2]*data_point_z
    return result
 
#=======================================================================================================================

# formally initialize theta vector and theta_0 vector
theta = [40,-15,-6000]
m =1

#=======================================================================================================================

# -1 labeled red point observations
x_no = np.array([158, 165, 170, 162, 176])
y_no = np.array([60, 72, 92, 65, 97])
z_no = np.array([1, 1, 1, 1, 1])
l_no = np.array([-1, -1, -1, -1, -1])
# +1 labeled red point observations
x_yes = [185, 194, 190, 197, 202]
y_yes = [82, 87, 92, 85, 95]
z_yes = [1, 1, 1, 1, 1]
l_yes = np.array([1, 1, 1, 1, 1])

# merge x-coordinates of all observations into 1 set
x = np.append(x_no, x_yes)
# merge y-coordinates of all observations into 1 set
y = np.append(y_no, y_yes)
# merge z-coordinates of all observations into 1 set
z = np.append(z_no, z_yes)
# merge z-coordinates of all observations into 1 set
labels = np.append(l_no, l_yes)


# iterate from 1 to m
for t in range(1, m+1):    
    # have we observed a change in theta vector
    is_theta_changed = False
    #iterate through data points
    for i in range(len(x)):
        print("\tObservation data point:["+str(x[i])+","+str(y[i])+","+str(z[i])+"]")
        print("\tLabel"+str(labels[i]))
        print("\tPrediction:"+str(prediction))
        print("\tFrom Theta:"+str(theta))
        plotPerceptronResults(theta, x[i], y[i], z[i])
        prediction = labels[i] * dotProduct(theta, x[i], y[i], z[i])
        
        if(prediction<=0):
            print("\tWrong Prediction")
            print("\tFrom Theta:"+str(theta))
            #our hypothesis prediction is wrong and has to change
            theta[0] += labels[i]*x[i]
            theta[1] += labels[i]*y[i]
            theta[2] += labels[i]*z[i]     
            is_theta_changed = True
            print("\tTo Theta:"+str(theta))
            print("-------------------------------------")
    if(is_theta_changed==False):
        print("Finished at:"+str(t))
        print("Theta:"+str(theta))
        plotPerceptronResults(theta, 0, 0, 0)
        break
    print("==============================================================")
