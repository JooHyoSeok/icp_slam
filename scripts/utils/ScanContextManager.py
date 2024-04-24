import asyncio
import numpy as np
from icecream import ic
import calc_distance_sc
np.set_printoptions(precision=4)

import time
from scipy import spatial

# def xy2theta(x, y):
#     if (x >= 0 and y >= 0): 
#         theta = 180/np.pi * np.arctan(y/x);
#     if (x < 0 and y >= 0): 
#         theta = 180 - ((180/np.pi) * np.arctan(y/(-x)));
#     if (x < 0 and y < 0): 
#         theta = 180 + ((180/np.pi) * np.arctan(y/x));
#     if ( x >= 0 and y < 0):
#         theta = 360 - ((180/np.pi) * np.arctan((-y)/x));

#     return theta


# def pt2rs(point, gap_ring, gap_sector, num_ring, num_sector):
#     x = point[0]
#     y = point[1]
#     # z = point[2]
    
#     if(x == 0.0):
#         x = 0.001
#     if(y == 0.0):
#         y = 0.001
    
#     theta = xy2theta(x, y)
#     faraway = np.sqrt(x*x + y*y)
    
#     idx_ring = np.divmod(faraway, gap_ring)[0]       
#     idx_sector = np.divmod(theta, gap_sector)[0]

#     if(idx_ring >= num_ring):
#         # print("overflow" , x , y)
#         idx_ring = num_ring-1 # python starts with 0 and ends with N-1
    
#     return int(idx_ring), int(idx_sector)


# def ptcloud2sc(ptcloud, sc_shape, max_length):
#     num_ring = sc_shape[0] # 20
#     num_sector = sc_shape[1] # 60

#     gap_ring = max_length/num_ring # 4 한 칸이 4m
#     gap_sector = 360/num_sector # 6 한 칸이 6degree
    
#     enough_large = 500 # 각 링과 섹터에 저장할 수 있는 데이터 포인트 수 제한
#     # ex. 2번 째 링과 3 섹터에 포함되는 데이터의 수가 500개가 넘어가면 안 됨.
#     # 메모리 관리, 성능 최적화, 데이터 품질 보장
#     sc_storage = np.zeros([enough_large, num_ring, num_sector])# shape (500,20,60)
#     # sc_storage 배열은 각 링(ring)과 섹터(sector) 조합에 대한 
#     # 데이터 포인트의 높이 값을 저장하는 3차원 배열. 
#     # 이 배열은 포인트 클라우드 데이터에서 각 포인트의 공간적 위치를 링과 섹터로 분류한 후,
#     # 해당 위치에 대응되는 높이 정보를 체계적으로 기록하는 데 사용
#     sc_counter = np.zeros([num_ring, num_sector])     # shape (20,60)
#     # ring row, sector col  data number
#     # sc_counter 배열은 각 링과 섹터 조합에 현재 저장된 데이터 포인트의 수

#     # ptcloud data shape (5000,3)
#     num_points = ptcloud.shape[0] # 5000
#     for pt_idx in range(num_points):
#         point = ptcloud[pt_idx, :] # x,y,z
#         point_height = point[2] + 2.0 # for setting ground is roughly zero 
#         idx_ring, idx_sector = pt2rs(point, gap_ring, gap_sector, num_ring, num_sector)
#         if sc_counter[idx_ring, idx_sector] >= enough_large:
#             continue
#         sc_storage[int(sc_counter[idx_ring, idx_sector]), idx_ring, idx_sector] = point_height
#         sc_counter[idx_ring, idx_sector] = sc_counter[idx_ring, idx_sector] + 1

#     sc = np.amax(sc_storage, axis=0)
#     # (20, 60) 링, 섹터 별 높이 값 최대 값 추출
        
#     return sc


# def sc2rk(sc):
#     # 링 별 평균값
#     # print(sc)
#     return np.mean(sc, axis=1) # (20,0)

# def distance_sc(sc1, sc2):
#     # sc1, sc2 shape (20, 60)
#     num_sectors = sc1.shape[1] # 60
#     sc1 = np.copy(sc1)
#     # repeate to move 1 columns
#     _one_step = 1 # const 1칸 이동
#     sim_for_each_cols = np.zeros(num_sectors) # 각 섹터 이동 후 계산된 유사도 저장하기 위한 배열
#     '''
#     코사인 유사도를 측정할 때 열을 오른쪽으로 이동(shift)시키면서 측정하는 이유는 스캔 컨텍스트의 회전 불변성을 확보하기 위해서입니다. 
#     이 접근법은 특히 로봇이나 센서가 같은 장소를 다른 각도에서 스캔했을 때 매우 유용합니다. 
#     다시 말해, 이동 또는 회전으로 인해 포인트 클라우드 데이터가 회전된 경우에도, 이를 효과적으로 매칭할 수 있도록 도와줍니다.
#     '''
#     for i in range(num_sectors): # 60
#         # Shift
#         sc1 = np.roll(sc1, _one_step, axis=1) #  columne shift
#         # 열 오른쪽으로 shift 맨 마지막 열은 첫번째로
#         #compare
#         sum_of_cossim = 0 # 코사인 유도 합 저장 변수
#         num_col_engaged = 0 # 유사도 계산에 사용된 섹터의 수를 저장한다.
#         for j in range(num_sectors):
#             col_j_1 = sc1[:, j]
#             col_j_2 = sc2[:, j]
#             # if (~np.any(col_j_1) or ~np.any(col_j_2)): 
#             #     # print("zero divide error")
#             #     # to avoid being divided by zero when calculating cosine similarity
#             #     # - but this part is quite slow in python, you can omit it.
#             #     continue 

#             cossim = np.dot(col_j_1, col_j_2) / (np.linalg.norm(col_j_1) * np.linalg.norm(col_j_2))
#             sum_of_cossim = sum_of_cossim + cossim
#             num_col_engaged += 1

#         # save 
#         sim_for_each_cols[i] = sum_of_cossim / num_col_engaged
#     yaw_diff = np.argmax(sim_for_each_cols) + 1 # because python starts with 0 
    
#     sim = np.max(sim_for_each_cols)
#     # print('yaw_diff', yaw_diff * 6.0 , "deg" , 'sim', sim)
#     cos_dist = 1 - sim

#     return cos_dist, yaw_diff

    
class ScanContextManager:
    def __init__(self, shape=[20,60], num_candidates=10, threshold=0.15): # defualt configs are same as the original paper 
        ##@param : 
        self.shape = shape 
        self.num_candidates = num_candidates
        self.threshold = threshold

        self.max_length = 80 # recommended but other (e.g., 100m) is also ok.

        self.ENOUGH_LARGE = 15000 # capable of up to ENOUGH_LARGE number of nodes 
        self.ptclouds = [None] * self.ENOUGH_LARGE
        self.scancontexts = [None] * self.ENOUGH_LARGE
        self.ringkeys = [None] * self.ENOUGH_LARGE

        self.curr_node_idx = 0
       

    def addNode(self, node_idx, ptcloud):
        '''  각 노드마다 ptcloud, scan context, ring key 저장
        내장 함수 : ptcloud2sc , sc2rk 
        '''
        
        # 각 노드마다 ptcloud, scan context, ring key 저장
        # sc = ptcloud2sc(ptcloud, self.shape, self.max_length) # (20, 60) 링, 섹터 별 높이 값 최대 값 추출
        # rk = sc2rk(sc) # 링, 섹터 중 높이 최대값 저장된 (20,60)에서 링 별 평균값
        
        # 10 times faster cpp 

        sc = calc_distance_sc.ptcloud2sc(ptcloud, self.shape, self.max_length)
        rk= calc_distance_sc.sc2rk(sc)

        self.curr_node_idx = node_idx
        self.ptclouds[node_idx] = ptcloud
        self.scancontexts[node_idx] = sc
        self.ringkeys[node_idx] = rk


    def getPtcloud(self, node_idx):
        
        return self.ptclouds[node_idx]



    def detectLoop(self):        
        exclude_recent_nodes = 30
        valid_recent_node_idx = self.curr_node_idx - exclude_recent_nodes
        if(valid_recent_node_idx < 1):
            return None, None, None
        else:
            # step 1
            ringkey_history = np.array(self.ringkeys[:valid_recent_node_idx])
            ringkey_tree = spatial.KDTree(ringkey_history)

            ringkey_query = self.ringkeys[self.curr_node_idx]
            _, nncandidates_idx = ringkey_tree.query(ringkey_query, k=self.num_candidates)
            ic()
            # step 2
            query_sc = self.scancontexts[self.curr_node_idx]
            
            nn_dist = 1.0 # initialize with the largest value of distance
            nn_idx = None
            nn_yawdiff = None
            for ith in range(self.num_candidates):
                candidate_idx = nncandidates_idx[ith]
                candidate_sc = self.scancontexts[candidate_idx]
                # print(candidate_sc.shape , query_sc.shape) -> (20, 60)
                # s = time.time() 
                # dist, yaw_diff = distance_sc(candidate_sc, query_sc)
                # print(f"py {time.time() - s}")
                # s = time.time()
                dist,yaw_diff = calc_distance_sc.distance_sc(candidate_sc, query_sc)
                # print(f"cpp {time.time() - s}")
                # print(f"{dist} {t1} {yaw_diff} {t2}")
                if(dist < nn_dist):
                    nn_dist = dist
                    nn_yawdiff = yaw_diff
                    nn_idx = candidate_idx

            print(f"best_loop_nn_dist : {nn_dist}")
            if(nn_dist < self.threshold):
                nn_yawdiff_deg = nn_yawdiff * (360/self.shape[1])
                ic("Loop Detection")
                print(f"nn_dist : {nn_dist}")
                return nn_idx, nn_dist, nn_yawdiff_deg # loop detected!
            else:
                return None, None, None
