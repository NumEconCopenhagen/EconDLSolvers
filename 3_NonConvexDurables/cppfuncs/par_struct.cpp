typedef struct par_struct
{
 bool full;
 int seed;
 int D;
 int T;
 double beta;
 double* d_ubar;
 double rho;
 double R;
 double nu;
 double* delta;
 double sigma_xi;
 double sigma_psi;
 double eta;
 int Nxi;
 int Npsi;
 double sigma_eps;
 double exp_noise_taste;
 int Nshocks;
 double mu_m0;
 double sigma_m0;
 double mu_p0;
 double sigma_p0;
 double mu_n10;
 double sigma_n10;
 double mu_n20;
 double sigma_n20;
 int cppthreads;
 int Nstates;
 int Nstates_pd;
 int Nactions;
 int Noutcomes;
 int NDC;
 double* kappa;
 double* omega;
 double* xi;
 double* xi_w;
 double* psi;
 double* psi_w;
} par_struct;

bool get_bool_par_struct(par_struct* x, char* name){

 if( strcmp(name,"full") == 0 ){ return x->full; }
 else {return false;}

}


int get_int_par_struct(par_struct* x, char* name){

 if( strcmp(name,"seed") == 0 ){ return x->seed; }
 else if( strcmp(name,"D") == 0 ){ return x->D; }
 else if( strcmp(name,"T") == 0 ){ return x->T; }
 else if( strcmp(name,"Nxi") == 0 ){ return x->Nxi; }
 else if( strcmp(name,"Npsi") == 0 ){ return x->Npsi; }
 else if( strcmp(name,"Nshocks") == 0 ){ return x->Nshocks; }
 else if( strcmp(name,"cppthreads") == 0 ){ return x->cppthreads; }
 else if( strcmp(name,"Nstates") == 0 ){ return x->Nstates; }
 else if( strcmp(name,"Nstates_pd") == 0 ){ return x->Nstates_pd; }
 else if( strcmp(name,"Nactions") == 0 ){ return x->Nactions; }
 else if( strcmp(name,"Noutcomes") == 0 ){ return x->Noutcomes; }
 else if( strcmp(name,"NDC") == 0 ){ return x->NDC; }
 else {return -9999;}

}


double get_double_par_struct(par_struct* x, char* name){

 if( strcmp(name,"beta") == 0 ){ return x->beta; }
 else if( strcmp(name,"rho") == 0 ){ return x->rho; }
 else if( strcmp(name,"R") == 0 ){ return x->R; }
 else if( strcmp(name,"nu") == 0 ){ return x->nu; }
 else if( strcmp(name,"sigma_xi") == 0 ){ return x->sigma_xi; }
 else if( strcmp(name,"sigma_psi") == 0 ){ return x->sigma_psi; }
 else if( strcmp(name,"eta") == 0 ){ return x->eta; }
 else if( strcmp(name,"sigma_eps") == 0 ){ return x->sigma_eps; }
 else if( strcmp(name,"exp_noise_taste") == 0 ){ return x->exp_noise_taste; }
 else if( strcmp(name,"mu_m0") == 0 ){ return x->mu_m0; }
 else if( strcmp(name,"sigma_m0") == 0 ){ return x->sigma_m0; }
 else if( strcmp(name,"mu_p0") == 0 ){ return x->mu_p0; }
 else if( strcmp(name,"sigma_p0") == 0 ){ return x->sigma_p0; }
 else if( strcmp(name,"mu_n10") == 0 ){ return x->mu_n10; }
 else if( strcmp(name,"sigma_n10") == 0 ){ return x->sigma_n10; }
 else if( strcmp(name,"mu_n20") == 0 ){ return x->mu_n20; }
 else if( strcmp(name,"sigma_n20") == 0 ){ return x->sigma_n20; }
 else {return NAN;}

}


double* get_double_p_par_struct(par_struct* x, char* name){

 if( strcmp(name,"d_ubar") == 0 ){ return x->d_ubar; }
 else if( strcmp(name,"delta") == 0 ){ return x->delta; }
 else if( strcmp(name,"kappa") == 0 ){ return x->kappa; }
 else if( strcmp(name,"omega") == 0 ){ return x->omega; }
 else if( strcmp(name,"xi") == 0 ){ return x->xi; }
 else if( strcmp(name,"xi_w") == 0 ){ return x->xi_w; }
 else if( strcmp(name,"psi") == 0 ){ return x->psi; }
 else if( strcmp(name,"psi_w") == 0 ){ return x->psi_w; }
 else {return NULL;}

}


