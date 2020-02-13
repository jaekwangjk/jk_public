//
//  simple_functions.h
//  Created by Kim, Jaekwang on 2/21/18.
//  List of functions included
//  [1] Scalar shear-rate, i.e. the second invariant of shear-rate tensor
//      at cylindrical Coordinate
//  [2] Moore Type Thixtoropic viscosity function


#ifndef simple_functions_h
#define simple_functions_h

#include <stdio.h>

using namespace dealii;

template <int dim>
double get_shear_rate(const SymmetricTensor<2,dim> symgrad_u, const double ur, const double r_point)
{
    return std::sqrt( 2.*(symgrad_u*symgrad_u + ur*ur/(r_point*r_point) ) );
}

double viscosity_Moore (const double lambda, const double eta_0, const double eta_str)  // (lambda, model parameter #1, model parameter #2 ... )
{

    double eta = eta_0 + eta_str * lambda;
    return eta;
}


#endif /* header_h */
