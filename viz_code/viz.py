from open3d import *    
import numpy as np

def main():
    bbox = read_point_cloud("dump/000003_pc.ply") # Read the point cloud
    pts = read_point_cloud("dump/000003_vgen_pc.ply") # Read the point cloud
    draw_geometries([bbox, pts]) # Visualize the point cloud     

if __name__ == "__main__":
    main()
