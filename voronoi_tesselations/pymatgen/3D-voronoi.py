import numpy as np 
import scipy as sp 
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay
from matplotlib import cm


data_points = np.random.randint(100,size=(10,2))
points = np.append(data_points, [[999,999], [-999,999], [999,-999], [-999,-999]], axis = 0)
# compute Voronoi tesselation
vor = Voronoi(points)
# plot
voronoi_plot_2d(vor)
# colorize
for region in vor.regions:
    if not -1 in region:
        polygon = [vor.vertices[i] for i in region]
        plt.fill(*zip(*polygon))
# fix the range of axes
plt.plot(data_points[:,0], data_points[:,1], 'ko')
plt.xlim(np.min(data_points[:,0])-1.0, np.max(data_points[:,0])+1.0)
plt.ylim(np.min(data_points[:,1])-1.0, np.max(data_points[:,1])+1.0)
plt.show()

points_3D=np.array([[0.0000000000000000, 0.0000000000000000,  0.0000000000000000],
[-0.0000000000000001,  1.9883854783306150,  1.9883854783306152],
[1.9883854783306150,  0.0000000000000000, 1.9883854783306152], 
[1.9883854783306147,  1.9883854783306150,  0.0000000000000002]])
points_3D = np.append(points_3D, [[999,999,999], [-999,999,999], [999,-999,999], [-999,-999,999],[999,999,-999], [-999,999,-999], [999,-999,-999], [-999,-999,-999]], axis = 0)
vor_3D=Voronoi(points_3D)
