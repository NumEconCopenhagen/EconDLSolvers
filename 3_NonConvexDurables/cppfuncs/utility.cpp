#ifndef MAIN
#define UTILITY
#include "header.cpp"
#endif

namespace utility {

double func(double c, double d, par_struct* par){ // utility function

    double utility_c = pow(c,par->alpha);
    double utility_d = pow(d+par->d_ubar,1.0-par->alpha);
    double cobb_d = utility_c*utility_d;
    double utility = pow(cobb_d,1.0-par->rho)/(1.0-par->rho);
    
    return utility;

}

} // namespace