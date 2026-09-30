import numpy as np
from collections import deque


class BoundaryVert:
    def __init__(self, _v_id, _theta):
        self.v_id = _v_id
        self.theta = _theta
        self.prev = None
        self.next = None

class Boundary:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def is_empty(self):
        return self.size == 0
    

    def append(self, v_id, theta):
        new_vert = BoundaryVert(v_id, theta)
        if self.is_empty():
            self.head = new_vert
            self.tail = new_vert
            new_vert.prev = new_vert
            new_vert.next = new_vert
        else:
            new_vert.prev = self.tail
            new_vert.next = self.head
            self.tail.next = new_vert
            self.head.prev = new_vert
            self.tail = new_vert
        self.size += 1

    def insert(self, target_v_id, v_id, theta):
        if self.is_empty():
            return
        current = self.head
        for _ in range(self.size):
            if current.v_id == target_v_id:
                new_vert = BoundaryVert(v_id, theta)
                new_vert.prev = current
                new_vert.next = current.next
                current.next.prev = new_vert
                current.next = new_vert
                if current == self.tail:
                    self.tail = new_vert
                self.size += 1
                return
            current = current.next

    def delete(self, vert):
        if self.is_empty():
            return
        if self.size == 1:
            self.head = None
            self.tail = None
            self.size = 0
            return
        if vert == self.head:
            self.head = vert.next
        if vert == self.tail:
            self.tail = vert.prev
        vert.prev.next = vert.next
        vert.next.prev = vert.prev
        self.size -= 1

    def delete_by_id(self, v_id):
        if self.is_empty():
            raise ValueError("BoundaryAdvancingFront is empty")
        
        current_node = self.head
        while True:
            if current_node.v_id == v_id:
                if self.size == 1:
                    self.head = None
                    self.tail = None
                else:
                    current_node.prev.next = current_node.next
                    current_node.next.prev = current_node.prev

                    if current_node == self.head:
                        self.head = current_node.next
                    if current_node == self.tail:
                        self.tail = current_node.prev
                
                self.size -= 1
                return
            
            current_node = current_node.next
            if current_node.v_id == self.head.v_id:
                break
        raise ValueError(f"Vertex ID {v_id} not found in BoundaryAdvancingFront")

    def find(self, v_id):
        if self.is_empty():
            return None
        current = self.head
        for _ in range(self.size):
            if current.v_id == v_id:
                return current
            current = current.next
        return None
    
    def update_theta(self, v_id, new_theta):
        vert = self.find(v_id)
        if vert:
            vert.theta = new_theta
        else:
            raise ValueError(f"Vertex ID {v_id} not found in BoundaryAdvancingFront")
    
    def update_id(self, old_v_id, new_v_id):
        vert = self.find(old_v_id)
        if vert:
            vert.v_id = new_v_id
        else:
            raise ValueError(f"Vertex ID {old_v_id} not found in BoundaryAdvancingFront")
        

    def get_smallest_theta_id(self):
        if self.is_empty():
            return None
        min_theta_vert = self.head
        current = self.head.next
        for _ in range(1, self.size):
            if current.theta < min_theta_vert.theta:
                min_theta_vert = current
            current = current.next
        return min_theta_vert.v_id, min_theta_vert.theta




class AdvancingFront:
    def __init__(self, _v_boundary):
        self.boundary = Boundary()
        self.v_boundary = _v_boundary


    def bisector(self, v1, v2):
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        bisector = v1 / norm1 + v2 / norm2
        return bisector / np.linalg.norm(bisector)
    
    def trisector(self, v1, v2):
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        sin = np.sin(angle)
        
        t1, t2 = 1.0 / 3, 2.0 / 3
        a = np.sin((1 - t1) * angle) / sin
        b = np.sin(t1 * angle) / sin
        c = np.sin((1 - t2) * angle) / sin
        d = np.sin(t2 * angle) / sin

        trisector1 = a * v1 + b * v2
        trisector1 = trisector1 / np.linalg.norm(trisector1)
        trisector2 = c * v1 + d * v2
        trisector2 = trisector2 / np.linalg.norm(trisector2)
        return trisector1, trisector2
    
    def get_surface_normal(self):
        assert self.boundary.size >= 3, "Boundary must have at least 3 vertices to compute surface normal."
        min_v_id = self.boundary.get_smallest_theta_id()
        v_pos = self.v_boundary[min_v_id]
        prev_v_pos = self.v_boundary[self.boundary.find(min_v_id).prev.v_id]
        next_v_pos = self.v_boundary[self.boundary.find(min_v_id).next.v_id]
        surface_normal = np.cross(v_pos - prev_v_pos, next_v_pos - v_pos)
        surface_normal /= np.linalg.norm(surface_normal)
        return surface_normal

    

    def triangulate(self):
        boundary = self.boundary
        surface_normal = self.get_surface_normal()

        threshold_bisector = 85 * np.pi / 180
        threshold_trisector = 135 * np.pi / 180
        alpha_new_vert = 1.0

        new_faces = []
        
        while boundary.size > 3:
            v_id, theta = boundary.get_smallest_theta_id()

            # case 1:
            # 1. update prev and next vert theta
            # 2. remove current vert
            # 3. add new face
            # 4. update all thetas
            if theta <= threshold_bisector:
                current_vert = boundary.find(v_id)

                prev_prev_v_id = current_vert.prev.prev.v_id
                prev_v_id      = current_vert.prev.v_id
                next_v_id      = current_vert.next.v_id
                next_next_v_id = current_vert.next.next.v_id
                
                prev_prev_v = self.v[prev_prev_v_id]
                prev_v = self.v[prev_v_id]
                next_v = self.v[next_v_id]
                next_next_v = self.v[next_next_v_id]

                new_prev_theta = self.boundary_theta(prev_prev_v, prev_v, next_v, surface_normal)
                boundary.update_theta(prev_v_id, new_prev_theta)
                new_next_theta = self.boundary_theta(prev_v, next_v, next_next_v, surface_normal)
                boundary.update_theta(next_v_id, new_next_theta)

                boundary.delete_by_id(v_id)

                new_faces.append([prev_v_id, v_id, next_v_id])

                # all_boundary_theta[prev_v_id] = new_prev_theta
                # all_boundary_theta[next_v_id] = new_next_theta
                # del all_boundary_theta[v_id]

            # case 2: 
            # 1. update current node v_id and theta
            # 2. update prev and next vert theta
            # 3. add new vertex
            # 4. add new faces
            # 5. update all thetas
            elif theta <= threshold_trisector:
                current_vert = boundary.find(v_id)
                v_pos = self.v[v_id]

                prev_prev_v_id = current_vert.prev.prev.v_id
                prev_v_id      = current_vert.prev.v_id
                next_v_id      = current_vert.next.v_id
                next_next_v_id = current_vert.next.next.v_id
                
                prev_prev_v = self.v[prev_prev_v_id]
                prev_v = self.v[prev_v_id]
                next_v = self.v[next_v_id]
                next_next_v = self.v[next_next_v_id]

                new_v_id = self.v.shape[0]
                bisector = self.bisector(prev_v - self.v[v_id], next_v - self.v[v_id])
                new_v_pos = v_pos + alpha_new_vert * np.linalg.norm(prev_v + next_v - 2 * v_pos) * bisector
                self.v = np.vstack((self.v, new_v_pos))
                new_v_theta = self.boundary_theta(prev_v, new_v_pos, next_v, surface_normal)

                boundary.update_theta(v_id, new_v_theta)
                boundary.update_id(v_id, new_v_id)

                new_prev_theta = self.boundary_theta(prev_prev_v, prev_v, new_v_pos, surface_normal)
                boundary.update_theta(prev_v_id, new_prev_theta)

                new_next_theta = self.boundary_theta(prev_v, new_v_pos, next_next_v, surface_normal)
                boundary.update_theta(next_v_id, new_next_theta)

                new_faces.append([prev_v_id, v_id, new_v_id])
                new_faces.append([v_id, next_v_id, new_v_id])

                # all_boundary_theta[new_v_id] = new_v_theta
                # all_boundary_theta[prev_v_id] = new_prev_theta
                # all_boundary_theta[next_v_id] = new_next_theta
                # del all_boundary_theta[v_id]

            # case 3: 
            # 1. update current node v_id1 and theta1
            # 2. insert node v_id2 and theta2
            # 3. update prev and next vert theta
            # 4. add new vertices
            # 5. add new faces
            # 6. update all thetas
            else:
                current_vert = boundary.find(v_id)
                v_pos = self.v[v_id]

                prev_prev_v_id = current_vert.prev.prev.v_id
                prev_v_id      = current_vert.prev.v_id
                next_v_id      = current_vert.next.v_id
                next_next_v_id = current_vert.next.next.v_id
                
                prev_prev_v = self.v[prev_prev_v_id]
                prev_v = self.v[prev_v_id]
                next_v = self.v[next_v_id]
                next_next_v = self.v[next_next_v_id]

                # 4
                new_v_id0 = self.v.shape[0]
                new_edge_len = np.linalg.norm(prev_v + next_v - 2 * v_pos)
                trisector0, trisector1 = self.trisector(prev_v - self.v[v_id], next_v - self.v[v_id])
                new_v_pos0 = v_pos + alpha_new_vert * new_edge_len * trisector0
                self.v = np.vstack((self.v, new_v_pos0))
                new_v_id1 = self.v.shape[0]
                new_v_pos1 = v_pos + alpha_new_vert * new_edge_len * trisector1
                self.v = np.vstack((self.v, new_v_pos1))

                new_v_theta0 = self.boundary_theta(prev_v, new_v_pos0, new_v_pos1, surface_normal)
                new_v_theta1 = self.boundary_theta(new_v_pos0, new_v_pos1, next_v, surface_normal)

                # 1, 2                boundary.update_theta(v_id, new_v_theta0)
                boundary.update_theta(v_id, new_v_theta0)
                boundary.update_id(v_id, new_v_id0)
                boundary.insert(new_v_id0, new_v_id1, new_v_theta1)

                # 3
                new_prev_theta = self.boundary_theta(prev_prev_v, prev_v, new_v_pos0, surface_normal)
                new_next_theta = self.boundary_theta(new_v_pos1, next_v, next_next_v, surface_normal)
                boundary.update_theta(prev_v_id, new_prev_theta)
                boundary.update_theta(next_v_id, new_next_theta)

                # 5
                new_faces.append([prev_v_id, v_id, new_v_id0])
                new_faces.append([new_v_id0, v_id, new_v_id1])
                new_faces.append([v_id, next_v_id, new_v_id1])

                # # 6
                # all_boundary_theta[prev_v_id] = new_prev_theta
                # all_boundary_theta[next_v_id] = new_next_theta
                # all_boundary_theta[new_v_id0] = new_v_theta0
                # all_boundary_theta[new_v_id1] = new_v_theta1
                # del all_boundary_theta[v_id]


            # print(boundary.size)
            # for i in range(len(new_faces)):
            #     plot.add_lines(
            #         self.v[new_faces[i][0]],
            #         self.v[new_faces[i][1]],
            #         shading={
            #             "line_color": "red",
            #             "line_width": 1
            #         }
            #     )
            #     plot.add_lines(
            #         self.v[new_faces[i][1]],
            #         self.v[new_faces[i][2]],
            #         shading={
            #             "line_color": "red",
            #             "line_width": 1
            #         }
            #     )
            #     plot.add_lines(
            #         self.v[new_faces[i][2]],
            #         self.v[new_faces[i][0]],
            #         shading={
            #             "line_color": "red",
            #             "line_width": 1
            #         }
            #     )
            # cur_new_f_plot_id = len(new_faces)
            # tmp_f = np.concatenate((self.f, np.array(new_faces)), axis=0)
            # mp.plot(self.v, tmp_f, np.ones((tmp_f.shape[0], 3)), shading={
            #     'wireframe': False
            # })
            # a = 1

        current_vert = boundary.head
        prev_v_id      = current_vert.prev.v_id
        next_v_id      = current_vert.next.v_id
        new_faces.append([prev_v_id, v_id, next_v_id])

        return np.array(new_faces, dtype=int)


