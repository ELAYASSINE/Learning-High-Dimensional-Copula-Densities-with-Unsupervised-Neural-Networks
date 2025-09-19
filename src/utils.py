import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# from tensorflow.keras.utils import *
from statsmodels.distributions.copula.api import (
    CopulaDistribution, GumbelCopula,ClaytonCopula,GaussianCopula,IndependenceCopula,FrankCopula)
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import kl_div
from keras.models import Sequential, Model
from keras.optimizers import Adam
from tensorflow.keras import backend as Kb  
from math import *
from keras.layers import LeakyReLU,Input, Dense
import random
from scipy.linalg import det, inv

import os
import matplotlib.pyplot as plt
from scipy.stats import uniform, entropy, wasserstein_distance
from scipy.special import erf
from math import *
from scipy.stats import norm
# To Suppress TensorFlow-specific logs and some other warning
import warnings
tf.get_logger().setLevel('ERROR')  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore", category=UserWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#&&&&&&&&&&&& Functions
def compute_kl_scipy(estimated_density,true_density):
    T=true_density
    E=estimated_density
    T /= np.sum(T)
    E /= np.sum(E)
    epsilon = 1e-10
    T = np.clip(T, epsilon, None)
    E = np.clip(E, epsilon, None)
    kl_valuess = kl_div(T, E)
    print(np.sum(kl_valuess))
    return np.sum(kl_valuess)
def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)

def reciprocal_loss(y_true, y_pred):
    epsilon = 1e-10  # Small constant to avoid division by zero
    product = y_true * y_pred
    reciprocal = tf.pow(product + epsilon, -1)
    return tf.reduce_mean(reciprocal)
def generate_gaussian_data(mean, cov, n_samples=10000):
    return np.random.multivariate_normal(mean, cov, size=n_samples)
def compute_gaussian_mi(cov_matrix):
    return -0.5 * np.log(det(cov_matrix))
def logsumexp_loss(y_true, y_pred):
    loss = tf.math.reduce_logsumexp(y_pred) - tf.math.log(tf.cast(tf.shape(y_true)[0], tf.float32))
    return loss
def my_binary_crossentropy(y_true, y_pred):
    return -tf.math.reduce_mean(tf.math.log(y_true) + tf.math.log(y_pred))

def my_binary_crossentropy(y_true, y_pred):
    return -tf.math.reduce_mean(tf.math.log(y_true) + tf.math.log(y_pred))
def plot_metrics_separately(total_loss, self_consistency):
    epochs = range(1, len(total_loss) + 1)                                                      
    # Plot total_loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, total_loss, label='Total Loss', color='blue', marker='o')
    plt.xlabel('Epochs')  
    plt.ylabel('Total Loss')
    plt.title('Total Loss per Epoch')
    plt.grid(True)
    plt.show()
    # Plot self_consistency 
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, self_consistency, label='Self Consistency', color='green', marker='x')
    plt.xlabel('Epochs')
    plt.ylabel('Self Consistency')
    plt.title('Self Consistency (Estimated Copula density for i i d)')
    plt.grid(True)
    plt.show()
def density_copula(u, v, model, theta):
    if model == 'Independence':
        s = np.ones_like(u)
    elif model == 'Gaussian':
        x = norm.ppf(u)
        y = norm.ppf(v)
        s = (1 / np.sqrt(1 - theta**2)) * np.exp(((-theta**2) * (x**2 + y**2) + 2 * theta * x * y) / (2 * (1 - theta**2)))
    elif model == 'FGM':
        s = 1 + theta * (1 - 2 * u) * (1 - 2 * v)
    elif model == 'AMH':
        num = 1 + theta * ((1 + u) * (1 + v) - 3) + theta**2 * (1 - u) * (1 - v)
        den = (1 - theta * (1 - u) * (1 - v))**3
        s = num / den
    elif model == 'Clayton':
        # theta=-0.91
        s = (1 + theta) * ((u * v)**(-theta - 1)) * ((u**(-theta) + v**(-theta) - 1)**(-2 - 1/theta))
    elif model == 'Frank':
        num = theta * (1 - np.exp(-theta)) * np.exp(-theta * (u + v))
        den = ((1 - np.exp(-theta)) - (np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1))**2
        s = num / den
    elif model == 'Gumbel':
        A = np.log(u) * np.log(v)
        B = (-np.log(u))**theta + (-np.log(v))**theta
        s = (A**(theta - 1) * (B**(1 / theta) + theta - 1) * B**(-2 + 1 / theta) * np.exp(-B**(1 / theta))) / (u * v)
    else:
        raise ValueError(f"Unrecognized copula type: '{model}'")
    return s
def sample_amh_copula(theta, size=2000):
    samples = []
    for _ in range(size):
        u1 = np.random.uniform()
        t = np.random.uniform()
        a = 1 - u1
        b = 1 - theta * (1 + 2 * a * t) + 2 * theta**2 * a**2 * t
        c = 1 + theta * (2 - 4 * a + 4 * a * t) + theta**2 * (1 - 4 * a * t + 4 * a**2 * t)
        u2 = (2 * t * (a * theta - 1)**2) / (b + np.sqrt(c))
        samples.append([u1, u2])
    return np.array(samples)
def fgm_copula_sample(theta, size=2000):
    if not (-1 <= theta <= 1):
        raise ValueError("Theta must be in [-1, 1] for the FGM copula.")
    u = np.random.uniform(0, 1, size)
    w = np.random.uniform(0, 1, size)
    v = w + theta * u * (1 - u) * (1 - 2 * w)
    return np.column_stack([u,v])
def independence_copula_sample(size=2000):
    u1 = np.random.uniform(0, 1, size)
    u2 = np.random.uniform(0, 1, size)
    return u1, u2

def plot_copula_densities(y_true, yhat, test_data, title):
    y_true = np.asarray(y_true).flatten()
    yhat = np.asarray(yhat).flatten()
    test_data = np.asarray(test_data)
    if test_data.ndim != 2 or test_data.shape[1] != 2:
        raise ValueError("test_data must have shape (n_samples, 2)")
    if y_true.ndim != 1 or y_true.shape[0] != test_data.shape[0]:
        raise ValueError("y_true must have the same number of elements as test_data rows")
    if yhat.ndim != 1 or yhat.shape[0] != test_data.shape[0]:
        raise ValueError("yhat must have the same number of elements as test_data rows")
    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    grid_points = np.vstack([grid_x.flatten(), grid_y.flatten()]).T
    grid_z_true = griddata(test_data, y_true, grid_points, method='linear', fill_value=0).reshape(grid_x.shape)
    grid_z_est = griddata(test_data, yhat, grid_points, method='linear', fill_value=0).reshape(grid_x.shape)
    fig = plt.figure(figsize=(18,6), dpi=300)  
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(grid_x, grid_y, grid_z_true, cmap='viridis', edgecolor='none', alpha=0.8)
    ax1.set_title(f"{title} - True Copula Density", pad=20)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(grid_x, grid_y, grid_z_est, cmap='viridis', edgecolor='none', alpha=0.8)
    ax2.dist = 2  
    ax2.set_title(f"{title} - Estimated Copula Density", pad=20)
    plt.tight_layout()
    plt.show()
def compute_copula_metrics(estimated_density,true_density,kde_density):
    true_density = np.asarray(true_density).flatten()
    estimated_density = np.asarray(estimated_density).flatten()
    epsilon = 1e-10
    true_density = np.clip(true_density, epsilon, None)
    estimated_density = np.clip(estimated_density, epsilon, None)
    true_density /= np.sum(true_density)
    estimated_density /= np.sum(estimated_density)
    kl_values = kl_div(true_density, estimated_density)
    kl_divergence = np.sum(kl_values)
    kde_density=kde_density/np.sum(kde_density)
    kl_values_kde = kl_div(true_density, kde_density)
    kl_divergence_kde= np.sum(kl_values_kde)
    nll = np.mean(-np.log(estimated_density))  
    re_norm_1o = np.linalg.norm(true_density - estimated_density, ord=1) / np.linalg.norm(true_density, ord=1)
    re_norm_2o = np.linalg.norm(true_density - estimated_density, ord=2) / np.linalg.norm(true_density, ord=2)
    re_norm_info = np.linalg.norm(true_density - estimated_density, ord=np.inf) / np.linalg.norm(true_density, ord=np.inf)
    re_norm_1 = np.linalg.norm(true_density - kde_density, ord=1) / np.linalg.norm(true_density, ord=1)
    re_norm_2 = np.linalg.norm(true_density - kde_density, ord=2) / np.linalg.norm(true_density, ord=2)
    re_norm_inf = np.linalg.norm(true_density - kde_density, ord=np.inf) / np.linalg.norm(true_density, ord=np.inf)
    return {
        'KL': kl_divergence,
        'KL_kde':kl_divergence_kde,
        'NLL': nll,
        'RE_1_our': re_norm_1o,
        'RE_2_our': re_norm_2o,
        'RE_inf_our': re_norm_info,
        'RE_1_kde': re_norm_1,
        'RE_2_kde': re_norm_2,
        'RE_inf_kde': re_norm_inf
    }
def kl_divergence_plot(kl_histories,copula_name,rho):
    plt.figure(figsize=(12, 8))
    epochs = list(kl_histories.keys())
    kl_values = list(kl_histories.values())
    plt.plot(epochs, kl_values, label=f"KL Divergence (rho = {rho})", marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('KL Divergence')
    plt.title(f'Kullback-Leibler Divergence Over Epochs for {copula_name} (rho = {rho})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
def RE1(kl_histories,copula_name,rho):
    plt.figure(figsize=(12, 8))
    epochs = list(kl_histories.keys())
    kl_values = list(kl_histories.values())
    plt.plot(epochs, kl_values, label=f"RE1(rho = {rho})", marker='x')
    plt.xlabel('Epochs')
    plt.ylabel('Relative Error 1')
    plt.title(f'Relative Error 1 Over Epochs for {copula_name} (rho = {rho})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
def RE2(kl_histories,copula_name,rho):
    plt.figure(figsize=(12, 8))
    epochs = list(kl_histories.keys())
    kl_values = list(kl_histories.values())
    plt.plot(epochs, kl_values, label=f"RE2(rho = {rho})", marker='*')
    plt.xlabel('Epochs')
    plt.ylabel('Relative Error 2')
    plt.title(f'Relative Error 2 Over Epochs for {copula_name} (rho = {rho})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
def RE_infini(kl_histories,copula_name,rho):
    plt.figure(figsize=(12, 8))
    epochs = list(kl_histories.keys())
    kl_values = list(kl_histories.values())
    plt.plot(epochs, kl_values, label=f"RE1(rho = {rho})", marker='+')
    plt.xlabel('Epochs')
    plt.ylabel('Relative Error inf')
    plt.title(f'Relative Error inf Over Epochs for {copula_name} (rho = {rho})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
################
def K(x):
    return np.exp(-x**2/2)/np.sqrt(2*np.pi)
def Fn(X, x):
    N = len(X)
    h = np.std(X) * N**(-1/5)
    return np.mean(K((x - X) / h))
def Fn_tri_vect(X):
    N = len(X)
    r = np.zeros(N)
    for i in range(N):
        r[i] = Fn(X, X[i])
    return r

def k_tri(x):
    r = (x + 1) * ((-1 <= x) & (x <= 0)) + (1 - x) * ((0 < x) & (x <= 1))
    return r
def kp_tri(x):
    r = 1 * ((-1 <= x) & (x <= 0)) - 1 * ((0 < x) & (x <= 1))
    return r
def K_tri(x):
    r = (0.5 * x**2 + x + 0.5) * ((-1 <= x) & (x < 0)) + (0.5 + x - 0.5 * x**2) * ((0 <= x) & (x <= 1)) + 1 * (x > 1)
    return r
def Fn_tri(X, x):
    N = len(X)
    h = np.std(X) * N**(-1/5)
    return np.mean(K_tri((x - X) / h))
def Fn_tri_vect(X):
    N = len(X)
    r = np.zeros(N)
    for i in range(N):
        r[i] = Fn_tri(X, X[i])
    return r
def c_n_tri(u, v, F1n, F2n):
    N = len(F1n)
    h1 = np.std(F1n) * N**(-1/5)
    h2 = np.std(F2n) * N**(-1/5)
    kernel_value = k_tri((u - F1n) / h1) * k_tri((v - F2n) / h2)
    r = np.mean(kernel_value) / (h1 * h2)
    return r
def generate_gaussian_data(mean, cov, n_samples=10000):
    return np.random.multivariate_normal(mean, cov, size=n_samples)
def c_n_vect(F1n, F2n):
    N = len(F1n)
    r = np.zeros(N)
    for i in range(N):
        r[i] = c_n_tri(F1n[i], F2n[i], F1n, F2n)
    return r