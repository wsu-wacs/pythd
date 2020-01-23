"""
Functionality for creating 2D test datasets for MAPPER

By Kyle Brown <brown.718@wright.edu>
"""
import numpy as np

class DatasetGenerator:
    def __init__(self):
        self.data = np.zeros((0,2))
    
    def _add_data(self, data):
        """Appends newly generated data"""
        self.data = np.concatenate((self.data, data), axis=0)
    
    def get(self):
        """Get the generated data"""
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
    
    def gaussian(self, center=[0.0, 0.0], sd=[1.0, 1.0], num_points=10):
        """Add samples from a 2-D Gaussian
        
        Parameters
        ----------
        center : iterable
            Mean of the Gaussian distribution
        sd : iterable
            Standard deviations of the Guassian along each axis
        num_points : int
            Number of points to sample
        """
        self._add_data(np.random.normal(loc=center, scale=sd, size=(num_points, 2)))
        return self
    
    def line(self, start=[0.0, 0.0], end=[1.0, 0.0], num_points=10, noise=0.0):
        """Add a line segment, with optional Gaussian noise
        
        Parameters
        ----------
        start : iterable
            Starting point of the line segment
        end : iterable
            Ending point of the line segment
        num_points : int
            Number of points in the line
        noise : float
            Standard deviation of optional Gaussian noise to add to the points of the line
            Set to 0 to disable noise
        """
        t = np.linspace(0.0, 1.0, num=num_points, endpoint=True)
        
        x = start[0] + (end[0] - start[0])*t + np.random.normal(scale=noise, size=t.shape)
        y = start[1] + (end[1] - start[1])*t + np.random.normal(scale=noise, size=t.shape)
        self._add_data(np.array([x, y]).T)
        return self
    
    def random_disk(self, center=[0.0, 0.0], min_radius=0.0, max_radius=1.0, num_points=10):
        """A disk with points sampled uniformly
        
        Parameters
        ----------
        center: iterable
            The center of the disk
        min_radius: float
            The minimum radius of the disk. Set to 0 if you want a completely filled-in disk
        max_radius: float
            The maximum radius of the disk (exclusive)
        num_points : int
            Total number of points to be sampled in the disk
        """
        center = np.array(center)
        r = np.random.uniform(low=min_radius, high=max_radius, size=num_points)
        theta = np.random.uniform(low=0.0, high=2.0*np.pi, size=num_points)
        
        x = center[0] + r*np.cos(theta)
        y = center[1] + r*np.sin(theta)
        self._add_data(np.array([x, y]).T)
        return self