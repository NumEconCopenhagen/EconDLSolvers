namespace linear_interp {

long long binary_search(long long imin, long long Nx, double *x, double xi)
{
    long long imid, half;

    // a. checks
    if(xi <= x[0]){
        return 0;
    } else if(xi >= x[Nx-2]) {
        return Nx-2;
    }

    // b. binary search
    half = Nx/2;
    while(half){
        imid = imin + half;
        if (xi > x[imid]) {
            imin = imid;
        }
        Nx -= half;
        half = Nx/2;
    }

    return imin;

}

////////
// 1d //
////////

double _interp_1d(double *grid1, long long Nx1, double *value, 
                  double xi1, long long j1)
{

    // a. left/right
    double nom_1_left = grid1[j1+1]-xi1;
    double nom_1_right = xi1-grid1[j1];

    // b. interpolation
    double denom = grid1[j1+1]-grid1[j1];
    double nom = 0;
    for(long long k1 = 0; k1 < 2; k1++){
        double nom_1 = k1 == 0 ? nom_1_left : nom_1_right;
        nom += nom_1*value[j1+k1];
    }
    return nom/denom;    

}

double interp_1d(double *grid1, long long Nx1, double *value, double xi1)
{

    // a. search in each dimension
    long long j1 = binary_search(0,Nx1,grid1,xi1);
    
    return _interp_1d(grid1,Nx1,value,xi1,j1);

}

long long *interp_1d_prep(long long Nyi)
{

    // a. search in non-last dimensions
    
    // b. prep
    long long *prep = new long long[Nyi];

    return prep;

}

void _interp_1d_mon_vec(long long *prep, double *grid1, long long Nx1, double *value,
                        double *xi1, double *yi, long long Nyi, bool monotone, bool search)
{

    // a. search in last dimension
    if(search){
        for(long long i = 0; i < Nyi; i++){
            if(monotone && i > 0){
                long long j1 = prep[i-1];
                while(xi1[i] >= grid1[j1+1] && j1 < Nx1-2){
                    j1 += 1;
                }
                prep[i] = j1;
            } else {
                prep[i] = binary_search(0,Nx1,grid1,xi1[i]);
            }
        }
    }

    // b. interpolation
    for(long long i = 0; i < Nyi; i++){
        yi[i] = 0.0; // initialize
        for(long long k1 = 0; k1 < 2; k1++){
            long long j1 = prep[i];
            double nom_1 = k1 == 0 ? grid1[j1+1]-xi1[i] : xi1[i]-grid1[j1];            
            yi[i] += nom_1*value[j1+k1];
        }
    }

    for(long long i = 0; i < Nyi; i++){
        long long j1 = prep[i];
        yi[i] /= (grid1[j1+1]-grid1[j1]);
    }

}

void interp_1d_vec_mon_rep(long long *prep, double *grid1, long long Nx1, double *value,
                           double *xi1, double *yi, long long Nyi)
{
    
    _interp_1d_mon_vec(prep,grid1,Nx1,value,xi1,yi,Nyi,true,false);

}

void interp_1d_vec_mon_noprep(double *grid1, long long Nx1, double *value,
                              double *xi1, double *yi, long long Nyi)
{
    
    long long *prep = interp_1d_prep(Nyi);
    _interp_1d_mon_vec(prep,grid1,Nx1,value,xi1,yi,Nyi,true,true);
    delete[] prep;

}

////////
// 2d //
////////

double _interp_2d(double *grid1, double *grid2, long long Nx1, long long Nx2, double *value, 
                    double xi1, double xi2, long long j1, long long j2)
{

    // a. left/right
    double nom_1_left = grid1[j1+1]-xi1;
    double nom_1_right = xi1-grid1[j1];

    double nom_2_left = grid2[j2+1]-xi2;
    double nom_2_right = xi2-grid2[j2];

    // b. interopolation
    double denom = (grid1[j1+1]-grid1[j1])*(grid2[j2+1]-grid2[j2]);
    double nom = 0;
    for(long long k1 = 0; k1 < 2; k1++){
        double nom_1 = k1 == 0 ? nom_1_left : nom_1_right;
        for(long long k2 = 0; k2 < 2; k2++){
            double nom_2 = k2 == 0 ? nom_2_left : nom_2_right;  
            long long index = index::d2(j1+k1,j2+k2,Nx1,Nx2);      
            nom += nom_1*nom_2*value[index];
    } }
    return nom/denom;

}

double interp_2d(double *grid1, double *grid2, long long Nx1, long long Nx2, double *value, 
                 double xi1, double xi2)
{

    long long j1 = binary_search(0,Nx1,grid1,xi1);
    long long j2 = binary_search(0,Nx2,grid2,xi2);
    
    return _interp_2d(grid1,grid2,Nx1,Nx2,value,xi1,xi2,j1,j2);

}

void interp_2d_vec(double *grid1, double *grid2, long long Nx1, long long Nx2, double *value,
                   double *xi1, double *xi2, double *yi, long long Nyi)
{
    for(long long i = 0; i < Nyi; i++){
        yi[i] = interp_2d(grid1,grid2,Nx1,Nx2,value,xi1[i],xi2[i]);
    }

}

long long *interp_2d_prep(double *grid1, long long Nx1, double xi1, long long Nyi)
{

    // a. search in non-last dimensions
    long long j1 = binary_search(0,Nx1,grid1,xi1);
    
    // b. prep
    long long *prep = new long long[1+Nyi];
    prep[Nyi+0] = j1;

    return prep;

}

double interp_2d_only_last(long long *prep, double *grid1, double *grid2, long long Nx1, long long Nx2, double *value,
                           double xi1, double xi2, long long Nyi)
{

    // a. search in last dimension
    long long j1 = prep[0];
    long long j2 = binary_search(0,Nx2,grid2,xi2);
    
    return _interp_2d(grid1,grid2,Nx1,Nx2,value,xi1,xi2,j1,j2);

}

void _interp_2d_only_last_vec(long long *prep, double *grid1, double *grid2, long long Nx1, long long Nx2, double *value,
                              double xi1, double *xi2, double *yi, long long Nyi, bool monotone, bool search)
{

    // a. search in last dimension
    long long j1 = prep[Nyi + 0];
    if(search){
        for(long long i = 0; i < Nyi; i++){
            if(monotone && i > 0){
                long long j2 = prep[i-1];
                while(xi2[i] >= grid2[j2+1] && j2 < Nx2-2){
                    j2 += 1;
                }
                prep[i] = j2;
            } else {
                prep[i] = binary_search(0,Nx2,grid2,xi2[i]);
            }
        }
    }

    // b. initialize
    for(long long i = 0; i < Nyi; i++){
        yi[i] = 0.0;
    }

    // c. interpolation
    double nom_1_left = grid1[j1+1]-xi1;
    double nom_1_right = xi1-grid1[j1];

    double denom = (grid1[j1+1]-grid1[j1]);
    for(long long k1 = 0; k1 < 2; k1++){
        double nom_1 = k1 == 0 ? nom_1_left : nom_1_right;
        for(long long i = 0; i < Nyi; i++){
            for(long long k2 = 0; k2 < 2; k2++){
                long long j2 = prep[i];
                double nom_2 = k2 == 0 ? grid2[j2+1]-xi2[i] : xi2[i]-grid2[j2];      
                long long index = index::d2(j1+k1,j2+k2,Nx1,Nx2);      
                yi[i] += nom_1*nom_2*value[index];
            }
        }
    }

    for(long long i = 0; i < Nyi; i++){
        long long j2 = prep[i];
        yi[i] /= denom*(grid2[j2+1]-grid2[j2]);
    }

}

void interp_2d_only_last_vec(long long *prep, double *grid1, double *grid2, long long Nx1, long long Nx2, double *value,
                             double xi1, double *xi2, double *yi, long long Nyi)
{
    _interp_2d_only_last_vec(prep,grid1,grid2,Nx1,Nx2,value,xi1,xi2,yi,Nyi,false,true);
}

void interp_2d_only_last_vec_mon(long long *prep, double *grid1, double *grid2, long long Nx1, long long Nx2, double *value,
                                double xi1, double *xi2, double *yi, long long Nyi)
{
    _interp_2d_only_last_vec(prep,grid1,grid2,Nx1,Nx2,value,xi1,xi2,yi,Nyi,true,true);
}

void interp_2d_only_last_vec_mon_rep(long long *prep, double *grid1, double *grid2, long long Nx1, long long Nx2, double *value,
                                     double xi1, double *xi2, double *yi, long long Nyi)
{
    _interp_2d_only_last_vec(prep,grid1,grid2,Nx1,Nx2,value,xi1,xi2,yi,Nyi,true,false);
}

////////
// 3d //
////////


double _interp_3d(double *grid1, double *grid2, double *grid3, long long Nx1, long long Nx2, long long Nx3, double *value, 
                 double xi1, double xi2, double xi3, long long j1, long long j2, long long j3)
{

    // a. left/right    
    double nom_1_left = grid1[j1+1]-xi1;
    double nom_1_right = xi1-grid1[j1];

    double nom_2_left = grid2[j2+1]-xi2;
    double nom_2_right = xi2-grid2[j2];

    double nom_3_left = grid3[j3+1]-xi3;
    double nom_3_right = xi3-grid3[j3];

    // b. interpolation
    double denom = (grid1[j1+1]-grid1[j1])*(grid2[j2+1]-grid2[j2])*(grid3[j3+1]-grid3[j3]);
    double nom = 0;
    for(long long k1 = 0; k1 < 2; k1++){
        double nom_1 = k1 == 0 ? nom_1_left : nom_1_right;
        for(long long k2 = 0; k2 < 2; k2++){
            double nom_2 = k2 == 0 ? nom_2_left : nom_2_right;       
            for(long long k3 = 0; k3 < 2; k3++){
                double nom_3 = k3 == 0 ? nom_3_left : nom_3_right;
                long long index = index::d3(j1+k1,j2+k2,j3+k3,Nx1,Nx2,Nx3);  
                nom += nom_1*nom_2*nom_3*value[index];
    } } } 
    return nom/denom;

}

double interp_3d(double *grid1, double *grid2, double *grid3, long long Nx1, long long Nx2, long long Nx3, double *value, 
                 double xi1, double xi2, double xi3)
{

    long long j1 = binary_search(0,Nx1,grid1,xi1);
    long long j2 = binary_search(0,Nx2,grid2,xi2);
    long long j3 = binary_search(0,Nx3,grid3,xi3);
    
    return _interp_3d(grid1,grid2,grid3,Nx1,Nx2,Nx3,value,xi1,xi2,xi3,j1,j2,j3);

}

void interp_3d_vec(double *grid1, double *grid2, double *grid3, long long Nx1, long long Nx2, long long Nx3, double *value,
                   double *xi1, double *xi2, double *xi3, double *yi, long long Nyi)
{
    for(long long i = 0; i < Nyi; i++){
        yi[i] = interp_3d(grid1,grid2,grid3,Nx1,Nx2,Nx3,value,xi1[i],xi2[i],xi3[i]);
    }

}

long long *interp_3d_prep(double *grid1, double *grid2, long long Nx1, long long Nx2, double xi1, double xi2, long long Nyi)
{

    // a. search in non-last dimensions
    long long j1 = binary_search(0,Nx1,grid1,xi1);
    long long j2 = binary_search(0,Nx2,grid2,xi2);
    
    // b. prep
    long long *prep = new long long[2+Nyi];
    prep[Nyi+0] = j1;
    prep[Nyi+1] = j2;

    return prep;

}

double interp_3d_only_last(long long *prep, double *grid1, double *grid2, double *grid3, long long Nx1, long long Nx2, long long Nx3, double *value,
                           double xi1, double xi2, double xi3, long long Nyi)
{

    // a. search in last dimension
    long long j1 = prep[0];
    long long j2 = prep[1];
    long long j3 = binary_search(0,Nx3,grid3,xi3);
    
    return _interp_3d(grid1,grid2,grid3,Nx1,Nx2,Nx3,value,xi1,xi2,xi3,j1,j2,j3);

}

void _interp_3d_only_last_vec(long long *prep, double *grid1, double *grid2, double *grid3, long long Nx1, long long Nx2, long long Nx3, double *value,
                              double xi1, double xi2, double *xi3, double *yi, long long Nyi, bool monotone, bool search)
{

    // a. search in last dimension
    long long j1 = prep[Nyi + 0];
    long long j2 = prep[Nyi + 1];
    if(search){
        for(long long i = 0; i < Nyi; i++){
            if(monotone && i > 0){
                long long j3 = prep[i-1];
                while(xi3[i] >= grid3[j3+1] && j3 < Nx3-2){
                    j3 += 1;
                }
                prep[i] = j3;
            } else {
                prep[i] = binary_search(0,Nx3,grid3,xi3[i]);
            }
        }
    }

    // b. initialize
    for(long long i = 0; i < Nyi; i++){
        yi[i] = 0.0;
    }

    // c. interpolation
    double nom_1_left = grid1[j1+1]-xi1;
    double nom_1_right = xi1-grid1[j1];

    double nom_2_left = grid2[j2+1]-xi2;
    double nom_2_right = xi2-grid2[j2];

    double denom = (grid1[j1+1]-grid1[j1])*(grid2[j2+1]-grid2[j2]);
    for(long long k1 = 0; k1 < 2; k1++){
        double nom_1 = k1 == 0 ? nom_1_left : nom_1_right;
        for(long long k2 = 0; k2 < 2; k2++){
            double nom_2 = k2 == 0 ? nom_2_left : nom_2_right;        
            for(long long i = 0; i < Nyi; i++){
                for(long long k3 = 0; k3 < 2; k3++){
                    long long j3 = prep[i];
                    double nom_3 = k3 == 0 ? grid3[j3+1]-xi3[i] : xi3[i]-grid3[j3];   
                    long long index = index::d3(j1+k1,j2+k2,j3+k3,Nx1,Nx2,Nx3);         
                    yi[i] += nom_1*nom_2*nom_3*value[index];
                }
            }
        }
    }

    for(long long i = 0; i < Nyi; i++){
        long long j3 = prep[i];
        yi[i] /= denom*(grid3[j3+1]-grid3[j3]);
    }

}

void interp_3d_only_last_vec(long long *prep, double *grid1, double *grid2, double *grid3, long long Nx1, long long Nx2, long long Nx3, double *value,
                             double xi1, double xi2, double *xi3, double *yi, long long Nyi)
{
    _interp_3d_only_last_vec(prep,grid1,grid2,grid3,Nx1,Nx2,Nx3,value,xi1,xi2,xi3,yi,Nyi,false,true);
}

void interp_3d_only_last_vec_mon(long long *prep, double *grid1, double *grid2, double *grid3, long long Nx1, long long Nx2, long long Nx3, double *value,
                                double xi1, double xi2, double *xi3, double *yi, long long Nyi)
{
    _interp_3d_only_last_vec(prep,grid1,grid2,grid3,Nx1,Nx2,Nx3,value,xi1,xi2,xi3,yi,Nyi,true,true);
}

void interp_3d_only_last_vec_mon_rep(long long *prep, double *grid1, double *grid2, double *grid3, long long Nx1, long long Nx2, long long Nx3, double *value,
                                     double xi1, double xi2, double *xi3, double *yi, long long Nyi)
{
    _interp_3d_only_last_vec(prep,grid1,grid2,grid3,Nx1,Nx2,Nx3,value,xi1,xi2,xi3,yi,Nyi,true,false);
}

////////
// 4d //
////////


double _interp_4d(double *grid1, double *grid2, double *grid3, double *grid4, long long Nx1, long long Nx2, long long Nx3, long long Nx4, double *value, 
                 double xi1, double xi2, double xi3, double xi4, long long j1, long long j2, long long j3, long long j4)
{

    // a. left/right    
    double nom_1_left = grid1[j1+1]-xi1;
    double nom_1_right = xi1-grid1[j1];

    double nom_2_left = grid2[j2+1]-xi2;
    double nom_2_right = xi2-grid2[j2];

    double nom_3_left = grid3[j3+1]-xi3;
    double nom_3_right = xi3-grid3[j3];

    double nom_4_left = grid4[j4+1]-xi4;
    double nom_4_right = xi4-grid4[j4];

    // b. interpolation
    double denom = (grid1[j1+1]-grid1[j1])*(grid2[j2+1]-grid2[j2])*(grid3[j3+1]-grid3[j3])*(grid4[j4+1]-grid4[j4]);
    double nom = 0;
    for(long long k1 = 0; k1 < 2; k1++){
        double nom_1 = k1 == 0 ? nom_1_left : nom_1_right;
        for(long long k2 = 0; k2 < 2; k2++){
            double nom_2 = k2 == 0 ? nom_2_left : nom_2_right;       
            for(long long k3 = 0; k3 < 2; k3++){
                double nom_3 = k3 == 0 ? nom_3_left : nom_3_right;
                for(long long k4 = 0; k4 < 2; k4++){
                    double nom_4 = k4 == 0 ? nom_4_left : nom_4_right;
                    long long index = index::d4(j1+k1,j2+k2,j3+k3,j4+k4,Nx1,Nx2,Nx3,Nx4);  
                    nom += nom_1*nom_2*nom_3*nom_4*value[index];
    } } } }
    return nom/denom;

}

double interp_4d(double *grid1, double *grid2, double *grid3, double *grid4, long long Nx1, long long Nx2, long long Nx3, long long Nx4, double *value, 
                 double xi1, double xi2, double xi3, double xi4)
{

    long long j1 = binary_search(0,Nx1,grid1,xi1);
    long long j2 = binary_search(0,Nx2,grid2,xi2);
    long long j3 = binary_search(0,Nx3,grid3,xi3);
    long long j4 = binary_search(0,Nx4,grid4,xi4);
    
    return _interp_4d(grid1,grid2,grid3,grid4,Nx1,Nx2,Nx3,Nx4,value,xi1,xi2,xi3,xi4,j1,j2,j3,j4);

}


void interp_4d_vec(double *grid1, double *grid2, double *grid3, double *grid4, long long Nx1, long long Nx2, long long Nx3, long long Nx4, double *value,
                   double *xi1, double *xi2, double *xi3, double *xi4, double *yi, long long Nyi)
{
    for(long long i = 0; i < Nyi; i++){
        yi[i] = interp_4d(grid1,grid2,grid3,grid4,Nx1,Nx2,Nx3,Nx4,value,xi1[i],xi2[i],xi3[i],xi4[i]);
    }

}

long long *interp_4d_prep(double *grid1, double *grid2, double *grid3, long long Nx1, long long Nx2, long long Nx3, double xi1, double xi2, double xi3, long long Nyi)
{

    // a. search in non-last dimensions
    long long j1 = binary_search(0,Nx1,grid1,xi1);
    long long j2 = binary_search(0,Nx2,grid2,xi2);
    long long j3 = binary_search(0,Nx3,grid3,xi3);
    
    // b. prep
    long long *prep = new long long[3+Nyi];
    prep[Nyi+0] = j1;
    prep[Nyi+1] = j2;
    prep[Nyi+2] = j3;

    return prep;

}

double interp_4d_only_last(long long *prep, double *grid1, double *grid2, double *grid3, double *grid4, long long Nx1, long long Nx2, long long Nx3, long long Nx4, double *value,
                           double xi1, double xi2, double xi3, double xi4, long long Nyi)
{

    // a. search in last dimension
    long long j1 = prep[0];
    long long j2 = prep[1];
    long long j3 = prep[2];
    long long j4 = binary_search(0,Nx4,grid4,xi4);
    
    return _interp_4d(grid1,grid2,grid3,grid4,Nx1,Nx2,Nx3,Nx4,value,xi1,xi2,xi3,xi4,j1,j2,j3,j4);

}

void _interp_4d_only_last_vec(long long *prep, double *grid1, double *grid2, double *grid3, double *grid4, long long Nx1, long long Nx2, long long Nx3, long long Nx4, double *value,
                              double xi1, double xi2, double xi3, double *xi4, double *yi, long long Nyi, bool monotone, bool search)
{

    // a. search in last dimension
    long long j1 = prep[Nyi + 0];
    long long j2 = prep[Nyi + 1];
    long long j3 = prep[Nyi + 2];
    if(search){
        for(long long i = 0; i < Nyi; i++){
            if(monotone && i > 0){
                long long j4 = prep[i-1];
                while(xi4[i] >= grid4[j4+1] && j4 < Nx4-2){
                    j4 += 1;
                }
                prep[i] = j4;
            } else {
                prep[i] = binary_search(0,Nx4,grid4,xi4[i]);
            }
        }
    }

    // b. initialize
    for(long long i = 0; i < Nyi; i++){
        yi[i] = 0.0;
    }

    // c. interpolation
    double nom_1_left = grid1[j1+1]-xi1;
    double nom_1_right = xi1-grid1[j1];

    double nom_2_left = grid2[j2+1]-xi2;
    double nom_2_right = xi2-grid2[j2];

    double nom_3_left = grid3[j3+1]-xi3;
    double nom_3_right = xi3-grid3[j3];

    double denom = (grid1[j1+1]-grid1[j1])*(grid2[j2+1]-grid2[j2])*(grid3[j3+1]-grid3[j3]);;
    for(long long k1 = 0; k1 < 2; k1++){
        double nom_1 = k1 == 0 ? nom_1_left : nom_1_right;
        for(long long k2 = 0; k2 < 2; k2++){
            double nom_2 = k2 == 0 ? nom_2_left : nom_2_right;        
            for(long long k3 = 0; k3 < 2; k3++){
                double nom_3 = k3 == 0 ? nom_3_left : nom_3_right;        
                for(long long i = 0; i < Nyi; i++){
                    for(long long k4 = 0; k4 < 2; k4++){
                        long long j4 = prep[i];
                        double nom_4 = k4 == 0 ? grid4[j4+1]-xi4[i] : xi4[i]-grid4[j4];   
                        long long index = index::d4(j1+k1,j2+k2,j3+k3,j4+k4,Nx1,Nx2,Nx3,Nx4);         
                        yi[i] += nom_1*nom_2*nom_3*nom_4*value[index];
                    }
                }
            }
        }
    }

    for(long long i = 0; i < Nyi; i++){
        long long j4 = prep[i];
        yi[i] /= denom*(grid4[j4+1]-grid4[j4]);
    }

}

void interp_4d_only_last_vec(long long *prep, double *grid1, double *grid2, double *grid3, double *grid4, long long Nx1, long long Nx2, long long Nx3, long long Nx4, double *value,
                             double xi1, double xi2, double xi3, double *xi4, double *yi, long long Nyi)
{
    _interp_4d_only_last_vec(prep,grid1,grid2,grid3,grid4,Nx1,Nx2,Nx3,Nx4,value,xi1,xi2,xi3,xi4,yi,Nyi,false,true);
}

void interp_4d_only_last_vec_mon(long long *prep, double *grid1, double *grid2, double *grid3, double *grid4, long long Nx1, long long Nx2, long long Nx3, long long Nx4, double *value,
                                double xi1, double xi2, double xi3, double *xi4, double *yi, long long Nyi)
{
    _interp_4d_only_last_vec(prep,grid1,grid2,grid3,grid4,Nx1,Nx2,Nx3,Nx4,value,xi1,xi2,xi3,xi4,yi,Nyi,true,true);
}

void interp_4d_only_last_vec_mon_rep(long long *prep, double *grid1, double *grid2, double *grid3, double *grid4, long long Nx1, long long Nx2, long long Nx3, long long Nx4, double *value,
                                     double xi1, double xi2, double xi3, double *xi4, double *yi, long long Nyi)
{
    _interp_4d_only_last_vec(prep,grid1,grid2,grid3,grid4,Nx1,Nx2,Nx3,Nx4,value,xi1,xi2,xi3,xi4,yi,Nyi,true,false);
}


////////
// 5d //
////////

double _interp_5d(double *grid1, double *grid2, double *grid3, double *grid4, double *grid5, long long Nx1, long long Nx2, long long Nx3, long long Nx4, long long Nx5, double *value, 
                 double xi1, double xi2, double xi3, double xi4, double xi5, long long j1, long long j2, long long j3, long long j4, long long j5)
{

    // a. left/right    
    double nom_1_left = grid1[j1+1]-xi1;
    double nom_1_right = xi1-grid1[j1];

    double nom_2_left = grid2[j2+1]-xi2;
    double nom_2_right = xi2-grid2[j2];

    double nom_3_left = grid3[j3+1]-xi3;
    double nom_3_right = xi3-grid3[j3];

    double nom_4_left = grid4[j4+1]-xi4;
    double nom_4_right = xi4-grid4[j4];

    double nom_5_left = grid5[j5+1]-xi5;
    double nom_5_right = xi5-grid5[j5];

    // b. interpolation
    double denom = (grid1[j1+1]-grid1[j1])*(grid2[j2+1]-grid2[j2])*(grid3[j3+1]-grid3[j3])*(grid4[j4+1]-grid4[j4])*(grid5[j5+1]-grid5[j5]);
    double nom = 0;
    for(long long k1 = 0; k1 < 2; k1++){
        double nom_1 = k1 == 0 ? nom_1_left : nom_1_right;
        for(long long k2 = 0; k2 < 2; k2++){
            double nom_2 = k2 == 0 ? nom_2_left : nom_2_right;       
            for(long long k3 = 0; k3 < 2; k3++){
                double nom_3 = k3 == 0 ? nom_3_left : nom_3_right;
                for(long long k4 = 0; k4 < 2; k4++){
                    double nom_4 = k4 == 0 ? nom_4_left : nom_4_right;
                    for(long long k5 = 0; k5 < 2; k5++){
                        double nom_5 = k5 == 0 ? nom_5_left : nom_5_right;
                        long long index = index::d5(j1+k1,j2+k2,j3+k3,j4+k4,j5+k5,Nx1,Nx2,Nx3,Nx4,Nx5);  
                        nom += nom_1*nom_2*nom_3*nom_4*nom_5*value[index];

    } } } }}
    return nom/denom;

}

double interp_5d(double *grid1, double *grid2, double *grid3, double *grid4, double *grid5, long long Nx1, long long Nx2, long long Nx3, long long Nx4, long long Nx5, double *value, 
                 double xi1, double xi2, double xi3, double xi4, double xi5)
{

    long long j1 = binary_search(0,Nx1,grid1,xi1);
    long long j2 = binary_search(0,Nx2,grid2,xi2);
    long long j3 = binary_search(0,Nx3,grid3,xi3);
    long long j4 = binary_search(0,Nx4,grid4,xi4);
    long long j5 = binary_search(0,Nx5,grid5,xi5);
    
    return _interp_5d(grid1,grid2,grid3,grid4,grid5,Nx1,Nx2,Nx3,Nx4,Nx5,value,xi1,xi2,xi3,xi4,xi5,j1,j2,j3,j4,j5);

}

} // namespace