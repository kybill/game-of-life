#include <cstdint>
extern "C" __global__ void check_true_cell(int len, int32_t *x_arr, int32_t *y_arr, int32_t *x_out, int32_t *y_out, int32_t *nb_ct) 
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < len) {
        int32_t x = x_arr[i];
        int32_t y = y_arr[i];
        int neighbor_count = 0;

        for (int a = 0; a < len; a++) {
            if (a == i) {
                continue;
            }
            int32_t x_t = x_arr[a];
            int32_t y_t = y_arr[a];
            if ((x_t <= x + 1) && (x_t >= x - 1) && (y_t <= y + 1) && (y_t >= y - 1)) {
                neighbor_count++;
            }
        }

        nb_ct[i] = neighbor_count;

        if (neighbor_count == 2 || neighbor_count == 3) {
            x_out[i] = x;
            y_out[i] = y;
        }
    }
}

extern "C" __global__ void check_false_cell(int len, int32_t *x_arr, int32_t *y_arr, int true_len, int32_t *true_x_arr, int32_t *true_y_arr, int32_t *x_out, int32_t *y_out, int32_t *nb_ct)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < len) {
        int32_t x = x_arr[i];
        int32_t y = y_arr[i];
        int neighbor_count = 0;

        for (int a = 0; a < true_len; a++) {
            int32_t x_t = true_x_arr[a];
            int32_t y_t = true_y_arr[a];
            if ((x_t <= x + 1) && (x_t >= x - 1) && (y_t <= y + 1) && (y_t >= y - 1)) {
                neighbor_count++;
            }
        }

        nb_ct[i] = neighbor_count;

        if (neighbor_count == 3) {
            x_out[i] = x;
            y_out[i] = y;
        }
    }
}
