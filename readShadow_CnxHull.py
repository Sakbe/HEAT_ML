import numpy as np
import glob
from scipy.spatial import ConvexHull

# Define the prism vertices
prism_vertices = np.array([
    [1570, 200, -1280], [1543, 200, -1300], [1560, 280, -1285],
    [1531, 275, -1300], [1720, 220, -1510], [1700, 220, -1532],
    [1710, 305, -1510], [1690, 305, -1530]
])

# Compute the convex hull of the prism vertices
hull = ConvexHull(prism_vertices)

# Function to check if a point is inside the convex hull
def point_inside_hull(point, hull):
    return (hull.equations[:, :-1] @ point + hull.equations[:, -1] <= 0).all()


# Process only the specified folders
dirs = [
    '/scratch/gpfs/dc2313/HEAT/data/sparc_000000_eq1212.txt_output',
    '/scratch/gpfs/dc2313/HEAT/data/sparc_000000_eq1209.txt_output'
]

dirs = glob.glob('/scratch/gpfs/dc2313/HEAT/data/sparc_000000_eq*')
dirs.sort()

ShadowMasks = []
XYZ = []

for d in dirs:
    csv_path = d + '/000001/shadowMask_all.csv'
    try:
        data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
        
        # Filter points inside the convex hull
        filtered_data = [point for point in data if point_inside_hull(point[:3], hull)]
        
        # Append filtered ShadowMasks to the list
        ShadowMasks.append([point[3] for point in filtered_data])
        

    except IOError:
        print(f"File not found or unable to read: {csv_path}")


#  XYZ coordinates
XYZ.extend([point[:3] for point in filtered_data])
        
# Convert lists to numpy arrays
XYZ = np.array(XYZ)
ShadowMasks = np.array(ShadowMasks)

# Save filtered ShadowMasks
np.savez("Filtered_ShadowMasks.npz",XYZ=XYZ,ShadowMasks=ShadowMasks)

# Print the shapes of the arrays
print("Shape of XYZ array:", XYZ.shape)
print("Shape of ShadowMasks array:", ShadowMasks.shape)
print("First row of XYZ:", XYZ[-1])
