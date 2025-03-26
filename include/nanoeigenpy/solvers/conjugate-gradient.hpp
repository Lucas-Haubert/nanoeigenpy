/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include "nanoeigenpy/solvers/iterative-solver-base.hpp"
#include <Eigen/IterativeLinearSolvers>

namespace nanoeigenpy {
namespace nb = nanobind;

template <typename ConjugateGradient>
struct ConjugateGradientVisitor
    : nb::def_visitor<ConjugateGradientVisitor<ConjugateGradient>> {
  using MatrixType = typename ConjugateGradient::MatrixType;

  template <typename... Ts>
  void execute(nb::class_<ConjugateGradient, Ts...>& cl) {
    using namespace nb::literals;
    cl.def(IterativeSolverVisitor<ConjugateGradient>())
        .def(nb::init<>(), "Default constructor.")
        .def(nb::init<MatrixType>(), "A"_a,
             "Initialize the solver with matrix A for further Ax=b solving.\n"
             "This constructor is a shortcut for the default constructor "
             "followed by a call to compute().");
  }

  static void expose(nb::module_& m, const char* name) {
    nb::class_<ConjugateGradient>(
        m, name, nb::is_final())  // is_final good to replace bp::noncopyable ?
        .def(ConjugateGradientVisitor<ConjugateGradient>())
        .def(IdVisitor());
  }
};

template <typename ConjugateGradient>
void exposeConjugateGradient(nb::module_& m, const char* name) {
  ConjugateGradientVisitor<ConjugateGradient>::expose(m, name);
}

}  // namespace nanoeigenpy
