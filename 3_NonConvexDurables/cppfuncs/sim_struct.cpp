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
 double* taste_shocks;
 double* DC;
 double* adj;
 double* c;
 double* d;
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
 else if( strcmp(name,"taste_shocks") == 0 ){ return x->taste_shocks; }
 else if( strcmp(name,"DC") == 0 ){ return x->DC; }
 else if( strcmp(name,"adj") == 0 ){ return x->adj; }
 else if( strcmp(name,"c") == 0 ){ return x->c; }
 else if( strcmp(name,"d") == 0 ){ return x->d; }
 else {return NULL;}

}


