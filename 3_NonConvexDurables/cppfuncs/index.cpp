namespace index {

long long d2(long long i_x1, long long i_x2, long long Nx1, long long Nx2){

    long long i = i_x1*Nx2
            + i_x2;

    return i;

} // d2

long long d3(long long i_x1, long long i_x2, long long i_x3, long long Nx1, long long Nx2, long long Nx3){

    long long i = i_x1*Nx2*Nx3
            + i_x2*Nx3
            + i_x3;

    return i;

} // d3

long long d4(long long i_x1, long long i_x2, long long i_x3, long long i_x4, long long Nx1, long long Nx2, long long Nx3, long long Nx4){

    long long i = i_x1*Nx2*Nx3*Nx4
            + i_x2*Nx3*Nx4
            + i_x3*Nx4
            + i_x4;

    return i;

} // d4

long long d5(long long i_x1, long long i_x2, long long i_x3, long long i_x4, long long i_x5, long long Nx1, long long Nx2, long long Nx3, long long Nx4, long long Nx5){

    long long i = i_x1*Nx2*Nx3*Nx4*Nx5
            + i_x2*Nx3*Nx4*Nx5
            + i_x3*Nx4*Nx5
            + i_x4*Nx5
            + i_x5;

    return i;

} // d5

long long d6(long long i_x1, long long i_x2, long long i_x3, long long i_x4, long long i_x5, long long i_x6, long long Nx1, long long Nx2, long long Nx3, long long Nx4, long long Nx5, long long Nx6){

    long long i = i_x1*Nx2*Nx3*Nx4*Nx5*Nx6
            + i_x2*Nx3*Nx4*Nx5*Nx6
            + i_x3*Nx4*Nx5*Nx6
            + i_x4*Nx5*Nx6
            + i_x5*Nx6
            + i_x6;

    return i;

} // d6

} // namespace