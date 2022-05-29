import numpy as np

def Gabor(sigma, theta, Lambda, phi, gamma, ksize=None):
    """Adopted from https://en.wikipedia.org/wiki/Gabor_filter"""
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Determine the kernel size
    if ksize is None:
        nstds = 3  # Number of standard deviation sigma
        xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
        xmax = np.ceil(max(1, xmax))
        ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
        ymax = np.ceil(max(1, ymax))
        xmin = -xmax
        ymin = -ymax
    else:
        xmax = ymax = ksize // 2
        xmin = ymin = -xmax
        
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + phi)
    return gb