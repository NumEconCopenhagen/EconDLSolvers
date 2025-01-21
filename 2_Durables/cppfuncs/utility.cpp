#ifndef MAIN
#define UTILITY
#include "header.cpp"
#endif

namespace utility {

double calc_omega_sum(par_struct *par){ // calculate sum of omega

    double omega_sum = 0.0;
    for(long long j = 0; j < par->D; j++){
        omega_sum += par->omega[j];
    }

    return omega_sum;

}

double func_d(double *d, par_struct *par){ // utility function for durable consumption

    double utility_d = 1.0;
    for(long long j = 0; j < par->D; j++){
        double djtot = d[j]+par->d_ubar[j];
        double dj_power = par->omega[j]*(1-par->rho);
        utility_d *= pow(djtot,dj_power);
    }

    return utility_d;
}

double func(double c, double *d, par_struct* par){ // utility function

    double utility_d = func_d(d, par);
    double omega_sum = calc_omega_sum(par);
    double utility_c = pow(c,(1-omega_sum)*(1.0-par->rho));
    
    return utility_c*utility_d/(1.0-par->rho);

}

double marg_func_c(double c, double *d, par_struct* par){ // marginal utility of consumption
    
    double utility_d = func_d(d, par);
    double omega_sum = calc_omega_sum(par);
    double c_power = (1-omega_sum)*(1-par->rho)-1.0;
    double factor = 1-omega_sum;

    return factor*pow(c,c_power)*utility_d;

} 

double inverse_marg_func_c(double mu, double *d,  par_struct* par){ // inverse of the marginal utility of consumption

    double utility_d = func_d(d, par);
    double omega_sum = calc_omega_sum(par);
    double c_power = (1-omega_sum)*(1-par->rho)-1.0;
    double factor = 1-omega_sum;

    return pow(mu/(factor*utility_d),1/c_power);

}

} // namespace