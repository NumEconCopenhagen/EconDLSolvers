#ifndef MAIN
#define BUDGET
#include "header.cpp"
#endif

namespace budget {

double adj_cost(double Delta, double tau){
    // adjustment cost function

    return tau*pow(Delta,2.0);

}

double d_consume_all(double tau, double epsilon, double m){
    // compute maximum investment level if all cash-on-hand spend except for epsilon

    double d_max;
    if (tau == 0.0){
    
        d_max = m-epsilon;
    
    } else{
      
        double a = (-tau);
        double b = -1;
        double c = m-epsilon;
        double denominator = 2*a;
        double numerator = (-b-sqrt(b*b-4*a*c));
        d_max = numerator/denominator;
    
    }


    return d_max;

}

double get_d_and_mbar(double m, double* n, double* choices, double* d, par_struct *par, egm_struct *egm){
    /// compute d and mbar from choices """

    double mbar = m;
    for(long long j = 0; j < par->D; j++){

        // a. minimum
        double dj_low = par->nonnegative_investment ? n[j] : 0.0;

        // b. maximum (upper bound enforced)
        double dj_high = MIN(n[j] + budget::d_consume_all(par->tau,0.0,mbar),egm->n_max);

        // c. choice
		d[j] = dj_low+(dj_high-dj_low)*choices[j];

        // d. update mbar
        double Delta_dj = d[j] - n[j];
		mbar = mbar - Delta_dj - budget::adj_cost(Delta_dj,par->tau);

    }

    return mbar;
    
}

} // namespace