// rls_filter.hpp
#pragma once
#include <vector>
#include <cmath>
#include <Eigen/Dense>
class RlsFilterCpp {
public:
    // Constructor
    RlsFilterCpp(
        const Eigen::MatrixXd& initial_w_transposed,
        int n_features,
        double lam,
        double halflife,
        double gamma,
        double epsilon
    ) : w(initial_w_transposed),
        num_assets(initial_w_transposed.rows()),
        n_features(initial_w_transposed.cols()),
        gamma_param(gamma),
        epsilon_param(epsilon)
    {
        beta_param = 1.0; // std::exp(std::log(0.5) / halflife);
        // Initialize the stack of P matrices
        P.reserve(num_assets);
        Eigen::MatrixXd p_initial = Eigen::MatrixXd::Identity(n_features, n_features) / lam;
        for (int i = 0; i < num_assets; ++i) {
            P.push_back(p_initial);
        }
    }
    // Predict method
    Eigen::VectorXd predict(const Eigen::VectorXd& x) {
        // (num_assets, n_features) * (n_features, 1) -> (num_assets, 1)
        return w * x;
    }
    // Update method: This is the performance-critical part
    void update(const Eigen::VectorXd& x, const Eigen::VectorXd& y_vector) {
        double C = gamma_param * (beta_param - 1) / beta_param;
        // This loop runs entirely in compiled C++ code
        for (int i = 0; i < num_assets; ++i) {
            // Get references/views to the specific asset's data
            Eigen::MatrixXd& Pi = P[i];
            Eigen::VectorXd wi = w.row(i);
            // --- RLS Calculations for asset i ---
            Eigen::VectorXd Px = Pi * x;
            double xPx = x.transpose() * Px;
            double r = 1.0 + xPx / beta_param;
            Eigen::VectorXd k = Px / (r * beta_param);
            double e = y_vector(i) - wi.dot(x);
            // --- L1 "extra" term calculation ---
            // Use Eigen's .array() for element-wise operations
            Eigen::ArrayXd S_vec_arr = wi.array().unaryExpr([](double val) {return static_cast<double>((val > 0) - (val < 0));});
            S_vec_arr /= (wi.array().abs() + epsilon_param);
            Eigen::VectorXd S_vec = S_vec_arr.matrix();
            Eigen::VectorXd PS = Pi * S_vec;
            double wPS_scalar = wi.dot(PS);
            Eigen::VectorXd extra = C * (PS - k * wPS_scalar);
            // --- Update state for asset i ---
            w.row(i) += (k * e + extra).transpose();
            Pi = (Pi - (k * k.transpose()) * r) / beta_param;
        }
    }
    // Public members for easy access from bindings
    Eigen::MatrixXd w;
    std::vector<Eigen::MatrixXd> P;
private:
    int num_assets;
    int n_features;
    double beta_param;
    double gamma_param;
    double epsilon_param;
};