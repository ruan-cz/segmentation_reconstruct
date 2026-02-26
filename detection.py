import numpy as np
import sklearn
import matplotlib.pyplot as plt






class Detection:
    def __init__(self, normals):
        self.normals = normals
        
        
    def detect_hole(self):
        # use mean shift to detect hole on sphere surface
        # transform normals to angle space
        
        normal_angle = np.zeros((self.normals.shape[0], 2))
        normal_angle[:, 0] = np.arccos(self.normals[:, 2])
        normal_angle[:, 1] = np.arctan2(self.normals[:, 1], self.normals[:, 0])
        
        # show
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(normal_angle[:, 0], normal_angle[:, 1])
        
        # do mean shift on angle space
        ms = sklearn.cluster.MeanShift()
        ms.fit(normal_angle)
        label = ms.labels_
        cluster_center = ms.cluster_centers_
        print(cluster_center)
        
        # show on fig
        ax.scatter(cluster_center[:, 0], cluster_center[:, 1], c='r')
        plt.show()
        

def sample_normals():
    def sample_n_normals(n, theta0, theta1, phi0, phi1):
    
        theta = np.random.uniform(theta0, theta1, n)
        phi = np.random.uniform(phi0, phi1, n)
        
        x0 = np.sin(theta) * np.cos(phi)
        y0 = np.sin(theta) * np.sin(phi)
        z0 = np.cos(theta)
        return np.stack((x0, y0, z0), axis=1)
    
    n0 = sample_n_normals(10, 0, 0.1*np.pi, 0, 0.2*np.pi)
    n1 = sample_n_normals(10, 0.5*np.pi, 0.6*np.pi, 0.5*np.pi, 0.6*np.pi)
    n2 = sample_n_normals(10, np.pi, 1.1*np.pi, 0, 0.2*np.pi)

    normals = np.concatenate((n0, n1, n2), axis=0)

    print(normals.shape)
    return normals


normals = sample_normals()
detection = Detection(normals)
detection.detect_hole()