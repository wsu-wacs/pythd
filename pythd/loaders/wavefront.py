"""
Code to load Wavefront OBJ files into a numpy array.

By Kyle Brown <brown.718@wright.edu>
"""
import numpy as np

class WavefrontOBJ:
    """
    Class representing information loaded from a Wavefront OBJ file.
    
    This contains information on vertices, vertex normals, texture coordinates, and faces.
    Since the purpose is to load the file as a dataset, it does not support any material 
    definitions or external texture files.
    
    Parameters
    ----------
    fname : str
        The name of the OBJ file to load. Ignored if a file object is passed.
    f : file object
        A file object to load the OBJ file from. If not given, a filename must be given in fname
    swapyz : bool
        Whether to swap Y and Z coordinates in vertices and normals when loading them in
    normalize : bool
        Whether to ensure normal vectors are normalized to be unit vectors.
    
    Attributes
    ----------
    vertices : numpy.ndarray
        The geometric vertices
    num_vertices : int
        The number of vertices
    normals : numpy.ndarray
        The normal vectors
    num_normals : int
        The number of normal vectors
    texcoords : numpy.ndarray
        The texture coordinates
    """
    def __init__(self, fname=None, f=None, swapyz=False, normalize=False):
        if f is None:
            with open(fname, "r") as f:
                self._load(f, swapyz=swapyz, normalize=normalize)
        elif fname is not None:
            self._load(f, swapyz=swapyz, normalize=normalize)
        else:
            raise ValueError("Either a filename or file object must be given.")
    
    def _process_vertices(self, v, swapyz):
        """Process a geometric vertex line in an OBJ file"""
        v = list(map(float, v))
        if swapyz:
            tmp = v[1]
            v[1] = v[2]
            v[2] = tmp
        # Add missing coordinates
        if len(v) == 3:
            v.append(1.0)
        return v
    
    def _process_normals(self, v, swapyz):
        """Process a normal vector line in an OBJ file"""
        v = list(map(float, v))
        if swapyz:
            tmp = v[1]
            v[1] = v[2]
            v[2] = tmp
        return v
        
    def _process_texcoords(self, v):
        """Process a texture coordinate line in an OBJ file"""
        v = list(map(float, v))
        # Add missing coordinates
        if len(v) == 1:
            v.append(0.0)
        if len(v) == 2:
            v.append(0.0)
        return v

    def _load(self, f, swapyz, normalize):
        """Load the OBJ file from a file object."""
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        self.vertex_dim = 0
        self.normal_dim = 0
        self.texcoord_dim = 0

        for line in f:
            line = line.strip()
            
            # Comments
            if line.startswith("#"):
                continue

            values = line.split()
            if not values:
                continue
            name = values[0].lower()
            
            # Geometric vertices
            if name == "v":
                self.vertex_dim = min(4, len(values[1:]))
                v = self._process_vertices(values[1:5], swapyz)
                self.vertices.append(v)
            # Normal vectors
            elif name == "vn":
                self.normal_dim = min(3, len(values[1:]))
                v = self._process_normals(values[1:4], swapyz)
                self.normals.append(v)
            # Texture coordinates
            elif name == "vt":
                self.texcoord_dim = min(3, len(values[1:]))
                v = self._process_texcoords(values[1:4])
                self.texcoords.append(v)
            # Faces
            elif name == "f":
                face_v = []
                face_vn = []
                face_vt = []
                
                for v in values[1:]:
                    w = v.split('/')
                    face_v.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        face_vt.append(int(w[1]))
                    else:
                        face_vt.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        face_vn.append(int(w[2]))
                    else:
                        face_vn.append(0)
                self.faces.append((face_v, face_vn, face_vt, None))
                    
        
        self.vertices = np.array(self.vertices)
        self.num_vertices = self.vertices.shape[0]
        
        self.normals = np.array(self.normals)
        self.num_normals = self.normals.shape[0]
        if normalize:
            self.normals = self.normals / np.linalg.norm(self.normals, ord=2, axis=1, keepdims=True)
        
        self.texcoords = np.array(self.texcoords)
        self.num_texcoords = self.texcoords.shape[0]

        self.num_faces = len(self.faces)
    
    def get_vertices(self):
        """Get the vertices in the OBJ file as a numpy array."""
        return self.vertices
    
    def get_num_vertices(self):
        """Get the number of geometric vertices in the OBJ file."""
        return self.num_vertices
    
    def get_vertex_dim(self):
        """Get the original dimension of vertices in the OBJ file."""
        return self.vertex_dim
    
    def get_normals(self):
        """Get the normal vectors in the OBJ file as a numpy array."""
        return self.normals
    
    def get_num_normals(self):
        """Get the number of normal vectors in the OBJ file."""
        return self.num_normals
    
    def get_normal_dim(self):
        """Get the original dimension of normal vectors in the OBJ file."""
        return self.normal_dim
    
    def get_texcoords(self):
        """Get the texture coordinates in the OBJ file as a numpy array."""
        return self.texcoords
    
    def get_num_texcoords(self):
        """Get the number of texture coordinates in the OBJ file"""
        return self.num_texcoords
    
    def get_texcoord_dim(self):
        """Get the original dimension of texture coordinates in the OBJ file."""
        return self.texcoord_dim
    
    def get_faces(self):
        """Get the faces in the OBJ file."""
        return self.faces
    
    def get_num_faces(self):
        """Get the number of faces in the OBJ file"""
        return self.num_faces