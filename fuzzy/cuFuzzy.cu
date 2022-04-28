// # CUDA acclerated fuzzy image processing operations
// # by ZHUO Xu (212138) @ SEU
// # https://github.com/z0gSh1u/seu-iau

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CLIP(v, l, r) ((v) <= (l) ? (l) : (v) >= (r) ? (r) : (v))
#define INDEX(r, c) ((r) * (w) + (c))

__device__ float muZeroD(int d, int sigma) {
  if (fabsf(d) > 2 * sigma) {
    return 0;
  }
  return expf(-d * d / (2 * sigma * sigma));
}

__device__ float muBlack(int z) {
  if (z > 180) {
    return 0;
  }
  return (180 - z) / 180;
}

__device__ float muWhite(int z) {
  if (z < 75) {
    return 0;
  }
  return (z - 75) / 180;
}

__global__ void fuzzyEdgeExtract3x3(int *res, int *img, int *params) {
  int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
  int h = params[0], w = params[1], sigma = params[2];
  if (i >= h || j >= w) {
    return;
  }

  // give up border processing
  int pad = 1; // pad = 3 // 2
  if (i < pad || i > h - pad || j < pad || j > w - pad) {
    return;
  }

  // collect patch
  int patch[9];
  for (int m = -pad, ptr = 0; m <= pad; m++) {
    for (int n = -pad; n <= pad; n++) {
      patch[ptr++] = img[INDEX(i + m, j + n)];
    }
  }

  // calculate diff
  int diff[9];
  for (int m = 0; m < 9; m++) {
    diff[m] = patch[m] - patch[4]; // minus z5
  }

  // calculate muZeroDs
  float muZeroDs[9];
  for (int m = 0; m < 9; m++) {
    muZeroDs[m] = muZeroD(diff[m], sigma); // diff to index
  }

  // enumerate z5
  float mu1Z5D2D6[256], mu2Z5D6D8[256], mu3Z5D8D4[256], mu4Z5D4D2[256], mu5Z5[256];
  for (int z5 = 0; z5 <= 255; z5++) {
    mu1Z5D2D6[z5] = MIN(MIN(muZeroDs[1], muZeroDs[5]), muWhite(z5));
    mu2Z5D6D8[z5] = MIN(MIN(muZeroDs[5], muZeroDs[7]), muWhite(z5));
    mu3Z5D8D4[z5] = MIN(MIN(muZeroDs[7], muZeroDs[3]), muWhite(z5));
    mu4Z5D4D2[z5] = MIN(MIN(muZeroDs[3], muZeroDs[1]), muWhite(z5));
    mu5Z5[z5] = muBlack(z5);
  }

  // calculate muZ5
  float muZ5[256];
  for (int z5 = 0; z5 <= 255; z5++) {
    muZ5[z5] =
        MAX(MAX(MAX(MAX(mu1Z5D2D6[z5], mu2Z5D6D8[z5]), mu3Z5D8D4[z5]), mu4Z5D4D2[z5]), mu5Z5[z5]);
  }

  // calculate the z5 to apply
  float sumMuZ5 = 0, sumZ5MuZ5 = 0;
  for (int z5 = 0; z5 <= 255; z5++) {
    sumZ5MuZ5 += muZ5[z5] * z5;
    sumMuZ5 += muZ5[z5];
  }

  // apply
  int z5Apply = roundf(sumZ5MuZ5 / sumMuZ5);
  z5Apply = CLIP(z5Apply, 0, 255);
  res[INDEX(i, j)] = z5Apply;
}
