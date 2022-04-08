// # CUDA acclerated morphology operations
// # by ZHUO Xu (212138) @ SEU
// # https://github.com/z0gSh1u/seu-iau

#define MIN(a, b) ((a) < (b) ? (a) : (b))

__global__ void binaryErode(int *res, const int *img, const int *se, const int *hw) {
  int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= hw[0] || j >= hw[1]) {
    return;
  }

  int hImg = hw[0], wImg = hw[1], hSE = hw[2], wSE = hw[3];
  int rhSE = hSE / 2, rwSE = wSE / 2; // radius
  int r, c, pix, match = 1;

  for (int m = -rhSE; m <= rhSE; m++) {
    for (int n = -rwSE; n <= rwSE; n++) {
      r = i + m;
      c = j + n;

      if (r < 0 || r >= hImg || c < 0 || c >= wImg) {
        // Keep original value for edges, give up processing.
        res[i * wImg + j] = img[i * wImg + j];
        return;
      } else {
        pix = img[r * wImg + c];
      }

      if (!pix && se[(m + rhSE) * wSE + (n + rwSE)]) {
        match = 0;
        break;
      }
    }
  }

  res[i * wImg + j] = match;
}

__global__ void grayscaleErode(int *res, const int *img, const int *se, const int *hw) {
  int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= hw[0] || j >= hw[1]) {
    return;
  }

  int hImg = hw[0], wImg = hw[1], hSE = hw[2], wSE = hw[3];
  int rhSE = hSE / 2, rwSE = wSE / 2; // radius
  int r, c, pix, min_ = 255;

  for (int m = -rhSE; m <= rhSE; m++) {
    for (int n = -rwSE; n <= rwSE; n++) {
      if (se[(m + rhSE) * wSE + (n + rwSE)]) {
        r = i + m;
        c = j + n;

        if (r < 0 || r >= hImg || c < 0 || c >= wImg) {
          pix = 255; // 255-pad
        } else {
          pix = img[r * wImg + c];
        }

        min_ = MIN(min_, pix);
      }
    }
  }

  res[i * wImg + j] = min_;
}