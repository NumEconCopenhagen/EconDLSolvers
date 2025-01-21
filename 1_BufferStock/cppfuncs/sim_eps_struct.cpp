typedef struct sim_eps_struct
{
 int seed;
 int N;
 double* states;
 double* states_pd;
 double* shocks;
 double* actions;
 double R;
 double* con;
 double* reward;
 double* MPC;
} sim_eps_struct;

int get_int_sim_eps_struct(sim_eps_struct* x, char* name){

 if( strcmp(name,"seed") == 0 ){ return x->seed; }
 else if( strcmp(name,"N") == 0 ){ return x->N; }
 else {return -9999;}

}


double* get_double_p_sim_eps_struct(sim_eps_struct* x, char* name){

 if( strcmp(name,"states") == 0 ){ return x->states; }
 else if( strcmp(name,"states_pd") == 0 ){ return x->states_pd; }
 else if( strcmp(name,"shocks") == 0 ){ return x->shocks; }
 else if( strcmp(name,"actions") == 0 ){ return x->actions; }
 else if( strcmp(name,"con") == 0 ){ return x->con; }
 else if( strcmp(name,"reward") == 0 ){ return x->reward; }
 else if( strcmp(name,"MPC") == 0 ){ return x->MPC; }
 else {return NULL;}

}


double get_double_sim_eps_struct(sim_eps_struct* x, char* name){

 if( strcmp(name,"R") == 0 ){ return x->R; }
 else {return NAN;}

}


