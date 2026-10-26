"""27-node sign-language skeleton (7 body + 10 per hand) and its adjacency.

The ``wlasl`` branch name is historical (WLASL lineage); it is the BdSL 27-keypoint layout.
Changing the topology also requires updating ``flip_index`` in ``feeders/feeder.py``.
"""
import sys

sys.path.extend(['../'])
from graph import tools




class Graph:
    def __init__(self, labeling_mode='spatial', graph='wlasl'):

        if graph == 'wlasl':
            num_node = 27
            self_link = [(i, i) for i in range(num_node)]
            inward_ori_index = [(5, 6), (5, 7),
                                (6, 8), (8, 10), (7, 9), (9, 11), 
                                (12,13),(12,14),(12,16),(12,18),(12,20),
                                (14,15),(16,17),(18,19),(20,21),
                                (22,23),(22,24),(22,26),(22,28),(22,30),
                                (24,25),(26,27),(28,29),(30,31),
                                (10,12),(11,22)]

            inward = [(i - 5, j - 5) for (i, j) in inward_ori_index]
            outward = [(j, i) for (i, j) in inward]
            neighbor = inward + outward
        elif graph  == 'kinetics':
            num_node = 18
            self_link = [(i, i) for i in range(num_node)]
            inward_ori_index = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12,11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
            inward = inward_ori_index
            outward = [(j, i) for (i, j) in inward]
            neighbor = inward + outward
        elif graph == 'ntu':
            num_node = 25
            self_link = [(i, i) for i in range(num_node)]
            inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                                (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                                (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                                (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
            inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
            outward = [(j, i) for (i, j) in inward]
            neighbor = inward + outward
        elif graph == 'coco17':
            # COCO-17 (HRNet 2D) topology, 0-indexed. 0 nose,1/2 eyes,3/4 ears,
            # 5/6 shoulders,7/8 elbows,9/10 wrists,11/12 hips,13/14 knees,15/16 ankles.
            # For the cross-domain demo (lab NTU vs in-the-wild HMDB/UCF, shared 17-joint topology).
            num_node = 17
            self_link = [(i, i) for i in range(num_node)]
            inward_ori_index = [(1, 0), (2, 0), (3, 1), (4, 2),
                                (5, 0), (6, 0), (5, 6), (5, 11), (6, 12), (11, 12),
                                (7, 5), (9, 7), (8, 6), (10, 8),
                                (13, 11), (15, 13), (14, 12), (16, 14)]
            inward = inward_ori_index
            outward = [(j, i) for (i, j) in inward]
            neighbor = inward + outward

        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(self.num_node, self.self_link, self.inward, self.outward)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # os.environ['DISPLAY'] = 'localhost:11.0'
    A = Graph('spatial').get_adjacency_matrix()
    for i in A:
        plt.imshow(i, cmap='gray')
        plt.show()
    print(A)
