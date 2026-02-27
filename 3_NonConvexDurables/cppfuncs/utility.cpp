#ifndef MAIN
#define UTILITY
#include "header.cpp"
#endif

namespace utility {

////////
// 1D //
////////

double func_1d(double c, double d, par_struct* par){ // utility function

    double utility_c = pow(c,1-par->omega[0]);
    double utility_d = pow(d+par->d_ubar[0],par->omega[0]);
    double cobb_d = utility_c*utility_d;
    double utility = pow(cobb_d,1.0-par->rho)/(1.0-par->rho);
    
    return utility;

}

double marg_u_c_func_1D(double c, double d, par_struct* par){

    double omega = par->omega[0];
    double rho   = par->rho;

    double one_minus_sum_omega = 1.0 - omega;

    // c exponent: (1 - omega)*(1 - rho) - 1
    double c_exp = one_minus_sum_omega * (1.0 - rho) - 1.0;
    double c_comp = std::pow(c, c_exp);

    // d exponent: omega*(1 - rho)
    double d_exp = omega * (1.0 - rho);
    double d_comp = std::pow(d + par->d_ubar[0], d_exp);

    double marg_u_c = one_minus_sum_omega * c_comp * d_comp;
    return marg_u_c;
}

double inv_marg_u_c_func_1D(double mu, double d, par_struct* par){

    double omega = par->omega[0];
    double rho   = par->rho;

    double omega_sum = omega;
    double one_minus_sum_omega = 1.0 - omega_sum;

    // prod_term = (d + d_ubar)^omega
    double prod_term = std::pow(d + par->d_ubar[0], omega);

    double denom = one_minus_sum_omega * std::pow(prod_term, 1.0 - rho);
    double exponent = 1.0 / (one_minus_sum_omega * (1.0 - rho) - 1.0);

    double c = std::pow(mu / denom, exponent);
    return c;
}

double logsumexp_1d(double v_keep, double v_adj, par_struct* par){

    double vmax = fmax(v_keep,v_adj);
    double logsumexp = vmax + par->sigma_eps*log(exp((v_keep-vmax)/par->sigma_eps) + exp((v_adj-vmax)/par->sigma_eps));

    return logsumexp;

}

////////
// 2D //
////////

double func_2d(double c, double d1, double d2, par_struct* par){ // utility function

    double utility_c = pow(c,1-par->omega[0]-par->omega[1]);
    double utility_d1 = pow(d1+par->d_ubar[0],par->omega[0]);
    double utility_d2 = pow(d2+par->d_ubar[1],par->omega[1]);
    double cobb_d = utility_c*utility_d1*utility_d2;
    double utility = pow(cobb_d,1.0-par->rho)/(1.0-par->rho);
    
    return utility;

}

double marg_u_c_func_2D(double c, double d1, double d2, par_struct* par){

    double omega1 = par->omega[0];
    double omega2 = par->omega[1];
    double rho    = par->rho;

    double omega_sum = omega1 + omega2;
    double one_minus_sum_omega = 1.0 - omega_sum;

    // c exponent: (1 - omega_sum)*(1 - rho) - 1
    double c_exp = one_minus_sum_omega * (1.0 - rho) - 1.0;
    double c_comp = std::pow(c, c_exp);

    // d1 exponent: omega1*(1 - rho)
    double d1_exp = omega1 * (1.0 - rho);
    double d1_comp = std::pow(d1 + par->d_ubar[0], d1_exp);

    // d2 exponent: omega2*(1 - rho)
    double d2_exp = omega2 * (1.0 - rho);
    double d2_comp = std::pow(d2 + par->d_ubar[1], d2_exp);

    double marg_u_c = one_minus_sum_omega * c_comp * d1_comp * d2_comp;
    return marg_u_c;
}

double inv_marg_u_c_func_2D(double mu, double d1, double d2, par_struct* par){

    double omega1 = par->omega[0];
    double omega2 = par->omega[1];
    double rho    = par->rho;

    double omega_sum = omega1 + omega2;
    double one_minus_sum_omega = 1.0 - omega_sum;

    // prod_term = (d1 + dbar1)^omega1 * (d2 + dbar2)^omega2
    double prod_term = std::pow(d1 + par->d_ubar[0], omega1)
                     * std::pow(d2 + par->d_ubar[1], omega2);

    double denom = one_minus_sum_omega * std::pow(prod_term, 1.0 - rho);
    double exponent = 1.0 / (one_minus_sum_omega * (1.0 - rho) - 1.0);

    double c = std::pow(mu / denom, exponent);
    return c;
}

double logsumexp_2d(double v_keep, double v_adj1, double v_adj2, double v_adj12,  par_struct* par){ // utility function

    double vmax = fmax(fmax(v_keep, v_adj1), fmax(v_adj2, v_adj12));
    double logsumexp = vmax + par->sigma_eps*log(
        exp((v_keep-vmax)/par->sigma_eps) 
        + exp((v_adj1-vmax)/par->sigma_eps) 
        + exp((v_adj2-vmax)/par->sigma_eps) 
        + exp((v_adj12-vmax)/par->sigma_eps)
    );

    return logsumexp;

}

} // namespace