"""
Functionality for creating 2D test datasets for MAPPER

By Kyle Brown <brown.718@wright.edu>
"""
import numpy as np

class DatasetGenerator:
    def __init__(self):
        self.data = np.zeros((0,2))
    
    def _add_data(self, data):
        self.data = np.concatenate((self.data, data), axis=0)
    
    def get(self):
        return self.data
    
    def circle(self, center=[0.0, 0.0], radius=1.0, num_points=20, noise=0.0):
        """Add a circle, with optional Gaussian noise
        
        Parameters
        ----------
        center : iterable
            Center of the circle
        radius : number
            Radius of the circle
        num_points : int
            Number of points in the circle
        noise : float
            Standard deviation of optional gaussian noise to add to the points.
            Set to 0 to disable noise.
        """
        delta = 2.0*np.pi / num_points
        theta = np.linspace(0.0, 2.0*np.pi, num=num_points, endpoint=False)
        center = np.array(center)
        
        x = center[0] + radius*np.cos(theta) + np.random.normal(scale=noise, size=theta.shape)
        y = center[1] + radius*np.sin(theta) + np.random.normal(scale=noise, size=theta.shape)
        self._add_data(np.array([x, y]).T)
        return self
    
    def line(self, start=[0.0, 0.0], end=[1.0, 0.0], num_points=10, noise=0.0):
        t = np.linspace(0.0, 1.0, num=num_points, endpoint=True)
        
        x = start[0] + (end[0] - start[0])*t + np.random.normal(scale=noise, size=t.shape)
        y = start[1] + (end[1] - start[1])*t + np.random.normal(scale=noise, size=t.shape)
        self._add_data(np.array([x, y]).T)
        return self
        