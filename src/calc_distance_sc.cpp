#include <iostream>
#include <string>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11//eigen.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <math.h>
#include <utility>


double xy2theta(double x , double y )
{
    double theta;

    if(x >= 0 && y >=0) theta = 180 / M_PI * atan(y/x);
    if (x < 0 and y >= 0) theta = 180 - ((180/M_PI) * atan(y/(-x)));
    if (x < 0 and y < 0) theta = 180 + ((180/M_PI) * atan(y/x));
    if ( x >= 0 and y < 0) theta = 360 - ((180/M_PI) * atan((-y)/x));

    return theta;
}
std::pair<int,int> pt2rs(const Eigen::VectorXd point , int gap_ring, int gap_sector, int num_ring, int num_sector)
{
    double x = point[0];
    double y = point[1];

    if(x == 0.0) x = 0.001;
    if(y == 0.0) y = 0.001;

    double theta = xy2theta(x,y);
    double faraway = sqrt(x*x + y*y);

    // Compute ring and sector indices
    int idx_ring = static_cast<int>(faraway / gap_ring);
    int idx_sector = static_cast<int>(theta / gap_sector);
    
    // Handle ring overflow
    if (idx_ring >= num_ring) {
        idx_ring = num_ring - 1; // Adjust index to stay within bounds
    }
    
    return std::make_pair(idx_ring, idx_sector);
}

Eigen::MatrixXd ptcloud2sc(const Eigen::MatrixXd& ptcloud , std::pair<int,int> sc_shape , int max_length)
{
    int num_ring = sc_shape.first;
    int num_sector = sc_shape.second;
    int gap_ring = max_length / num_ring ;
    int gap_sector = 360 / num_sector;
    int enough_large = 500 ;

    std::vector<std::vector<std::vector<double>>> sc_storage(enough_large, std::vector<std::vector<double>>(num_ring, std::vector<double>(num_sector, 0)));    
    Eigen::MatrixXd sc_counter = Eigen::MatrixXd::Zero(num_ring,num_sector);
    std::pair<int,int> idx;
    int num_points = ptcloud.rows();


    for(int pt_idx = 0 ; pt_idx < num_points ; pt_idx++)
    {
        Eigen::VectorXd point = ptcloud.row(pt_idx);
        double point_height = point(2) + 2.0 ;

        idx = pt2rs(point,gap_ring,gap_sector,num_ring,num_sector);
        int idx_ring = idx.first;
        int idx_sector = idx.second;

        if(sc_counter(idx_ring,idx_sector) >= enough_large) 
        {   
            continue;
        }

        sc_storage[int(sc_counter(idx_ring,idx_sector))][idx_ring][idx_sector] = point_height;
        sc_counter(idx_ring, idx_sector) = sc_counter(idx_ring, idx_sector) + 1;

    }

    Eigen::MatrixXd sc(num_ring, num_sector);
    for (int r = 0; r < num_ring; ++r) {
        for (int s = 0; s < num_sector; ++s) {
            double max_height = 0;
            for (int i = 0; i < sc_counter(r,s); ++i) {

                max_height = std::max(max_height, sc_storage[i][r][s]);
            }
            sc(r, s) = max_height;
        }
    }


    return sc;
}

Eigen::VectorXd sc2rk(Eigen::MatrixXd sc){

    Eigen::VectorXd row_means(sc.rows());
    for (int i = 0; i < sc.rows(); ++i) {
        row_means(i) = sc.row(i).mean();
    }

    return row_means;

}


std::pair<double, int> distance_sc(const Eigen::MatrixXd& sc1, const Eigen::MatrixXd& sc2) {
    int num_sectors = sc1.cols(); // 60
    Eigen::VectorXd sim_for_each_cols = Eigen::VectorXd::Zero(num_sectors);
    for (int i = 1; i < num_sectors; ++i) {
        // Create a copy of sc1 and shift it right by one column
       
        /*
        block() 함수의 사용법
        startRow: 블록이 시작하는 행의 인덱스
        startCol: 블록이 시작하는 열의 인덱스
        blockRows: 블록에 포함될 행의 수
        blockCols: 블록에 포함될 열의 수
        */

        Eigen::MatrixXd shifted_sc1 = sc1;
        
        // decrease
        shifted_sc1.block(0, i, sc1.rows(), shifted_sc1.cols() - i) = sc1.block(0, i-1, sc1.rows() , sc1.cols() - i);

        // increase
        shifted_sc1.block(0, 0, sc1.rows() , i) = sc1.block(0, num_sectors - i , sc1.rows() , i);

        // 0 , 1 ,59 , 1
        // 0,  2 , 58 , 2

        double sum_of_cossim = 0.0;
        int num_col_engaged = 0;

        for (int j = 0; j < num_sectors; ++j) {
            Eigen::VectorXd col_j_1 = shifted_sc1.col(j);
            Eigen::VectorXd col_j_2 = sc2.col(j);

            if (col_j_1.norm() == 0 || col_j_2.norm() == 0){
                continue;
            }

            double cossim = col_j_1.dot(col_j_2) / (col_j_1.norm() * col_j_2.norm());
            sum_of_cossim += cossim;
            num_col_engaged++;
        }

        // Avoid division by zero in case no columns were engaged

        sim_for_each_cols(i) = sum_of_cossim / num_col_engaged;
        
    }
    double max_value = 0.0;
    int idx = 0;

    for(int i = 0 ; i < 60 ; i ++) {

        if (sim_for_each_cols[i] > max_value){

            max_value = sim_for_each_cols[i];
            idx = i;
        }
    }

    int max_index;
    double sim = sim_for_each_cols.maxCoeff(&max_index);
    int yaw_diff = max_index; // because indexing starts at 0
    double cos_dist = 1 - sim;
    return {cos_dist, yaw_diff};
}

void test(){

    Eigen::Matrix4d mat;  // 4x4 행렬 생성
    mat << 1, 2, 3, 4,
           5, 6, 7, 8,
           9, 10, 11, 12,
           13, 14, 15, 16;

    // 중앙의 2x2 부분 행렬 추출
    Eigen::MatrixXd block = mat.block(1, 1, 2, 2);

    std::cout << "Original matrix:\n" << mat << std::endl;
    std::cout << "2x2 central block:\n" << block << std::endl;

}

PYBIND11_MODULE(calc_distance_sc, m) 
{
    m.doc() = "calc_distance_sc"; // optional module docstring
    m.def("distance_sc",&distance_sc,"distance_sc");
    m.def("pt2rs",&pt2rs,"pt2rs");
    m.def("ptcloud2sc",&ptcloud2sc,"ptcloud2sc");
    m.def("sc2rk",&sc2rk,"sc2rk");

    m.def("test",&test,"test");
}
