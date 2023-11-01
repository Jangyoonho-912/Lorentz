#%%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from Lorentz_function import *


#%%
axis_1 = 'x'
axis_2 = 'z'
N = 5500
plot_lorentz_attractor(axis_1, axis_2, N)
plot_xyz_time (N)

# %%
training_step = 100
V_x, V_y, V_z = voltage_transform ()
V_x = np.round(V_x[:training_step],3)


# %% 조건 별 처리 결과 확인용
V_d = 0
series = V_x
t_relax = 1e-3
process_result_plot (V_d, series, t_relax)
# %% make training set
input_size = 10
training_set_x, training_set_y, training_set_z = mk_Lr_training_set (t_relax, training_step,input_size)
#x-axis
training_set_0_x = np.array(training_set_x[0])
training_set_1_x = np.array(training_set_x[1])
training_set_2_x = np.array(training_set_x[2])
#y-axis
training_set_0_y = np.array(training_set_y[0])
training_set_1_y = np.array(training_set_y[1])
training_set_2_y = np.array(training_set_y[2])
#z-axis
training_set_0_z = np.array(training_set_z[0])
training_set_1_z = np.array(training_set_z[1])
training_set_2_z = np.array(training_set_z[2])
# %% x-axis
training_set = np.hstack((training_set_0_x, training_set_1_x, training_set_2_x))
update = 40
update_interval = 40
test_step = 5
y_tested, nrmse = Lr_training_test_update(training_set, training_step, test_step, t_relax, update, update_interval,input_size)

# %% XYZ_integrated
t_relax = 1e-3
training_step = 500
input_size = 150
update = 100
update_interval = 100
test_step = 5
#%%
training_set = mk_Lr_training_set_XYZ (t_relax, training_step,input_size)
#%%
y_tested = Lr_training_test_update_XYZ (training_set, training_step, test_step, t_relax, update, update_interval,input_size)

# %%
