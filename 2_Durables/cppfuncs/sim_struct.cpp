typedef struct sim_struct
{
 int N;
 int reps;
 double* states;
 double* states_pd;
 double* shocks;
 double* outcomes;
 double* actions;
 double* reward;
 double R;
 double* euler_error;
 double* MPC_c;
 double* MPC_d;
 double* R_transfer;
 double* R_transfers;
} sim_struct;

int get_int_sim_struct(sim_struct* x, char* name){

 if( strcmp(name,"N") == 0 ){ return x->N; }
 else if( strcmp(name,"reps") == 0 ){ return x->reps; }
 else {return -9999;}

}


double* get_double_p_sim_struct(sim_struct* x, char* name){

 if( strcmp(name,"states") == 0 ){ return x->states; }
 else if( strcmp(name,"states_pd") == 0 ){ return x->states_pd; }
 else if( strcmp(name,"shocks") == 0 ){ return x->shocks; }
 else if( strcmp(name,"outcomes") == 0 ){ return x->outcomes; }
 else if( strcmp(name,"actions") == 0 ){ return x->actions; }
 else if( strcmp(name,"reward") == 0 ){ return x->reward; }
 else if( strcmp(name,"euler_error") == 0 ){ return x->euler_error; }
 else if( strcmp(name,"MPC_c") == 0 ){ return x->MPC_c; }
 else if( strcmp(name,"MPC_d") == 0 ){ return x->MPC_d; }
 else if( strcmp(name,"R_transfer") == 0 ){ return x->R_transfer; }
 else if( strcmp(name,"R_transfers") == 0 ){ return x->R_transfers; }
 else {return NULL;}

}


double get_double_sim_struct(sim_struct* x, char* name){

 if( strcmp(name,"R") == 0 ){ return x->R; }
 else {return NAN;}

}


