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
