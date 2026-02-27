typedef struct sim_struct
{
 double R;
 int N;
 int reps;
 double* states;
 double* states_pd;
 double* shocks;
 double* outcomes;
 double* actions;
 double* reward;
 double* transfer_grid;
 int Ntransfer;
 double* R_transfer;
 double* individual_R_transfer;
 double* taste_shocks;
 double* DC;
 double* adj;
 double* c;
 double* d1;
 double* d2;
 double* euler_error_c;
} sim_struct;

double get_double_sim_struct(sim_struct* x, char* name){

 if( strcmp(name,"R") == 0 ){ return x->R; }
 else {return NAN;}

}


int get_int_sim_struct(sim_struct* x, char* name){

 if( strcmp(name,"N") == 0 ){ return x->N; }
 else if( strcmp(name,"reps") == 0 ){ return x->reps; }
 else if( strcmp(name,"Ntransfer") == 0 ){ return x->Ntransfer; }
 else {return -9999;}

}


double* get_double_p_sim_struct(sim_struct* x, char* name){

 if( strcmp(name,"states") == 0 ){ return x->states; }
 else if( strcmp(name,"states_pd") == 0 ){ return x->states_pd; }
 else if( strcmp(name,"shocks") == 0 ){ return x->shocks; }
 else if( strcmp(name,"outcomes") == 0 ){ return x->outcomes; }
 else if( strcmp(name,"actions") == 0 ){ return x->actions; }
 else if( strcmp(name,"reward") == 0 ){ return x->reward; }
 else if( strcmp(name,"transfer_grid") == 0 ){ return x->transfer_grid; }
 else if( strcmp(name,"R_transfer") == 0 ){ return x->R_transfer; }
 else if( strcmp(name,"individual_R_transfer") == 0 ){ return x->individual_R_transfer; }
 else if( strcmp(name,"taste_shocks") == 0 ){ return x->taste_shocks; }
 else if( strcmp(name,"DC") == 0 ){ return x->DC; }
 else if( strcmp(name,"adj") == 0 ){ return x->adj; }
 else if( strcmp(name,"c") == 0 ){ return x->c; }
 else if( strcmp(name,"d1") == 0 ){ return x->d1; }
 else if( strcmp(name,"d2") == 0 ){ return x->d2; }
 else if( strcmp(name,"euler_error_c") == 0 ){ return x->euler_error_c; }
 else {return NULL;}

}


