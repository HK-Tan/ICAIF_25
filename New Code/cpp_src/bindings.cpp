// bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h> // For std::vector
#include "rls_filter.hpp"
namespace py = pybind11;
// Helper function to convert std::vector<Eigen::MatrixXd> to a 3D NumPy array
py::array_t<double> get_p_as_numpy(const RlsFilterCpp& filter) {
    const auto& P_vec = filter.P;
    if (P_vec.empty()) {
        return py::array_t<double>();
    }
    size_t num_assets = P_vec.size();
    size_t n_features = P_vec[0].rows();
    // Create a 3D NumPy array
    py::array_t<double> result({num_assets, n_features, n_features});
    auto buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);
    // Copy data from each matrix in the vector to the NumPy array
    for (size_t i = 0; i < num_assets; ++i) {
        Eigen::Map<Eigen::MatrixXd>(ptr + i * n_features * n_features, n_features, n_features) = P_vec[i];
    }
    return result;
}
PYBIND11_MODULE(cpp_rls_filter, m) {
    m.doc() = "High-performance C++ implementation of the RLS filter";
    py::class_<RlsFilterCpp>(m, "CppExpL1L2Regression")
        .def(py::init<const Eigen::MatrixXd&, int, double, double, double, double>(),
             py::arg("initial_w"),
             py::arg("n_features"),
             py::arg("lam") = 0.1,
             py::arg("halflife") = 20.0,
             py::arg("gamma") = 0.01,
             py::arg("epsilon") = 1e-6,
             "Constructor for the C++ RLS Filter")
        .def("predict", &RlsFilterCpp::predict,
             py::arg("x"),
             "Predicts the output for all assets given an input vector x.")
        .def("update", &RlsFilterCpp::update,
             py::arg("x"), py::arg("y_vector"),
             "Updates the filter's weights for all assets.")
        // Expose w and P as read-only properties for inspection from Python
        .def_property_readonly("w", [](const RlsFilterCpp &self) {
            return self.w;
        })
        .def_property_readonly("P", &get_p_as_numpy);
}