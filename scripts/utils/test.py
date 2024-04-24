import numpy as np

# 간단한 스캔 컨텍스트 생성
sc1 = np.zeros((20, 60))  # 20링, 60섹터
sc2 = np.zeros((20, 60))

# sc1에 일부 데이터 삽입
sc1[5, :] = np.linspace(1, 0, 60)  # 5번째 링의 값을 1에서 0으로 선형 감소
sc1[10, :] = np.linspace(0, 1, 60)  # 10번째 링의 값을 0에서 1로 선형 증가

# sc2는 sc1을 15 열(90도) 이동
sc2[:, 15:60] = sc1[:, :45] + 1
sc2[:, :15] = sc1[:, 45:60] 

# distance_sc 함수 사용
def distance_sc(sc1, sc2):
    num_sectors = sc1.shape[1]
    _one_step = 1
    sim_for_each_cols = np.zeros(num_sectors)
    for i in range(num_sectors):
        sc1_rolled = np.roll(sc1, i, axis=1)
        sum_of_cossim = 0
        num_col_engaged = 0
        for j in range(num_sectors):
            col_j_1 = sc1_rolled[:, j]
            col_j_2 = sc2[:, j]
            if np.any(col_j_1) and np.any(col_j_2):
                cossim = np.dot(col_j_1, col_j_2) / (np.linalg.norm(col_j_1) * np.linalg.norm(col_j_2))
                sum_of_cossim += cossim
                num_col_engaged += 1

        if num_col_engaged > 0:
            sim_for_each_cols[i] = sum_of_cossim / num_col_engaged

    best_shift = np.argmax(sim_for_each_cols)
    best_sim = sim_for_each_cols[best_shift]
    return 1 - best_sim, best_shift

distance, shift = distance_sc(sc1, sc2)
print(f"Best match shift: {shift}, Distance: {distance}")


from icecream import ic, argumentToString
@argumentToString.register(np.ndarray)
def _(obj):
    return f"ndarray, shape={obj.shape}, dtype={obj.dtype}"


ic(sc1)