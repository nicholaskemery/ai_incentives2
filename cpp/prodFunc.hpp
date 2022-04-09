# pragma once

#include <Eigen/Eigen>
#include <vector>
#include <utility>


class ProdFunc {
public:
    ProdFunc(
        Eigen::ArrayXd A,
        Eigen::ArrayXd alpha,
        Eigen::ArrayXd B,
        Eigen::ArrayXd beta,
        Eigen::ArrayXd theta
    );

    std::tuple<double, double> f_single_i(
        int i,
        double Ks,
        double Kp
    ) const;

    std::tuple<Eigen::ArrayXd, Eigen::ArrayXd> f(
        const Eigen::ArrayXd& Ks,
        const Eigen::ArrayXd& Kp
    ) const;

    Eigen::Array22d jac_single_i(
        int i,
        const Eigen::ArrayXd& x
    ) const;

    Eigen::ArrayXd A;
    Eigen::ArrayXd B;
    Eigen::ArrayXd alpha;
    Eigen::ArrayXd beta;
    Eigen::ArrayXd theta;
};
