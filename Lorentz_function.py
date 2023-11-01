#%%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from function import *

# Define the Lorentz system of ODEs
def lorentz_system(t, Y, sigma, rho, beta):
    x, y, z = Y
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

def mk_lorentz_attractor():
    # Parameters
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0

    # Initial conditions
    initial_conditions = [1, 1, 1]

    # Time span
    t_span = (0, 1000)
    t_eval = np.linspace(t_span[0], t_span[1], 10000)

    # Solve the system of ODEs
    sol = solve_ivp(lorentz_system, t_span, initial_conditions, args=(sigma, rho, beta), t_eval=t_eval)

    # Extract the solution
    x = sol.y[0]
    y = sol.y[1]
    z = sol.y[2]
    x = x[500:10000]
    y = y[500:10000]
    z = z[500:10000]
    return x, y, z

def min_xyz():
    x, y, z = mk_lorentz_attractor()
    return min(x), min(y), min(z)

def max_xyz():
    x, y, z = mk_lorentz_attractor()
    return max(x), max(y), max(z)

# Plot the Lorentz attractor
def plot_lorentz_attractor(axis_1, axis_2, N):
    x, y, z = mk_lorentz_attractor()
    x_1 = x[:N]
    y_1 = y[:N]
    z_1 = z[:N]
    axis_dict = {'x': x_1, 'y': y_1, 'z': z_1}
    plt.figure(figsize=(10, 6))
    plt.plot(axis_dict[axis_1], axis_dict[axis_2], label='Lorentz Attractor', linewidth=0.5)
    plt.title('Lorentz Attractor')
    plt.xlabel(axis_1)
    plt.ylabel(axis_2)
    plt.grid()
    plt.legend()
    plt.show()

def plot_xyz_time (N):
    x, y, z = mk_lorentz_attractor()
    x_1 = x[:N]
    y_1 = y[:N]
    z_1 = z[:N]
    t = np.arange(0, N)
    plt.figure(dpi = 300)
    plt.plot(t,y_1,'g',label='y')
    plt.plot(t,z_1,'b',label='z')
    plt.plot(t,x_1,'r',label='x')
    plt.xlabel('Time step')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')
    plt.show()

def voltage_transform ():
    x, y, z = mk_lorentz_attractor()

    voltage_x = (x - min(x)) / (max(x) - min(x)) + 4
    voltage_y = (y - min(y)) / (max(y) - min(y)) + 4
    voltage_z = (z - min(z)) / (max(z) - min(z)) + 4

    return (voltage_x, voltage_y, voltage_z)

def mk_Lr_training_set (t_relax, training_step, input_size):
    xyz_cases = []
    voltage_xyz = voltage_transform ()
    for voltage_axis in voltage_xyz:
        v_axis = np.round(voltage_axis[:training_step],3)
        e = []
        # epoch = len(v)-input_size
        for i in range(len(v_axis)-input_size):
            e.append(v_axis[0+i:input_size+i])
        arr = np.array(e)

        V_d = [0, 1, 2]
        three_V_d_cases = []
        for vd in tqdm(V_d):
            processed_current = []
            for i in tqdm(range(len(arr))):
                processed_current.append(time_series_process (vd, arr[i], t_relax))
            three_V_d_cases.append(processed_current)
        xyz_cases.append(three_V_d_cases)

    return xyz_cases[0], xyz_cases[1], xyz_cases[2]

def mk_Lr_training_set_XYZ (t_relax, training_step, input_size):
    xyz_cases = []
    voltage_xyz = voltage_transform ()
    for voltage_axis in voltage_xyz:
        v_axis = np.round(voltage_axis[:training_step],3)
        e = []
        # epoch = len(v)-input_size
        for i in range(len(v_axis)-input_size):
            e.append(v_axis[0+i:input_size+i])
        arr = np.array(e)

        V_d = [0, 1, 2]
        three_V_d_cases = []
        for vd in tqdm(V_d):
            processed_current = []
            for i in tqdm(range(len(arr))):
                processed_current.append(time_series_process (vd, arr[i], t_relax))
            three_V_d_cases.append(processed_current)
        xyz_cases.append(three_V_d_cases)

    return np.hstack((np.hstack(xyz_cases[0]), np.hstack(xyz_cases[1]), np.hstack(xyz_cases[2])))

def Lr_training_test_update(training_set, training_step, test_step, t_relax, update, update_interval, input_size):    
    min_x, min_y, min_z = min_xyz()
    max_x, max_y, max_z = max_xyz()

    ###TRAINING###
    lr_x_1, lr_y_1, lr_z_1 = mk_lorentz_attractor()

    x_train = training_set
    y_train = lr_x_1[input_size:training_step].reshape(training_step-input_size, 1)
    mlr = LinearRegression()
    mlr.fit(x_train, y_train)

    x = np.arange(input_size+1, training_step+1)
    prediction = mlr.predict(training_set)
    plt.figure(dpi = 300, figsize=(10,3))
    plt.plot(x,y_train,'b',label='target')
    plt.scatter(x,prediction, label='pred')
    plt.xlabel('Time step')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper left')
    plt.show()

    ###TEST###
    y_test = []
    y_test_1 = lr_x_1[training_step-input_size:training_step]
    v_1 = (y_test_1 - min_x) / max_x + 4

    for i in tqdm(range(test_step)):
        new_input_0 = time_series_process (0, v_1, t_relax).reshape(1, -1)
        new_input_1 = time_series_process (1, v_1, t_relax).reshape(1, -1)
        new_input_2 = time_series_process (2, v_1, t_relax).reshape(1, -1)

        new_input = np.hstack((new_input_0, new_input_1, new_input_2))
    
        prediction = mlr.predict(new_input)
        y_test.append(prediction)
        y_test_1 = np.insert(y_test_1,input_size,prediction)
        y_test_1 = np.delete(y_test_1,0)
        if 0 <= i%update_interval <= update:
            y_test_1 = lr_x_1[training_step-input_size+i:training_step+i]
        v_1 = (y_test_1 - min_x) / (max_x - min_x) + 4
    
    y_predict = mlr.predict(x_train)
    y_tested = y_predict
    for i in range(len(y_test)):
        y_tested = np.vstack((y_tested, y_test[i]))

    x = np.arange(input_size+1, training_step+test_step+1)
    y_answer = lr_x_1[input_size:training_step+test_step].reshape(-1, 1)
    plt.figure(dpi = 300, figsize=(10,3))
    plt.plot(x,y_answer,'b',label='target')
    plt.plot(x,y_tested, 'r',label='pred')
    plt.xlabel('Time step')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper left')

    nrmse_train = calculate_rmse(y_answer[:training_step-input_size], y_tested[:training_step-input_size])
    nrmse = calculate_rmse_last_k(y_answer, y_tested, test_step)
    print(f'NRMSE for {training_step} training step: {nrmse_train}')
    print(f'NRMSE for {test_step} test step: {nrmse}')

    return y_tested, nrmse



def Lr_training_test_update_XYZ (training_set, training_step, test_step, t_relax, update, update_interval, input_size):    
    min_x, min_y, min_z = min_xyz()
    max_x, max_y, max_z = max_xyz()

    ###TRAINING###
    lr_x_1, lr_y_1, lr_z_1 = mk_lorentz_attractor()
    xyz_target = np.vstack((lr_x_1, lr_y_1, lr_z_1)).T
    
    x_train = training_set
    y_train = xyz_target[input_size:training_step].reshape(training_step-input_size, 3)
    mlr = LinearRegression()
    mlr.fit(x_train, y_train)

    x = np.arange(input_size+1, training_step+1)
    prediction = mlr.predict(training_set)
    plt.figure(dpi = 300, figsize=(10,3))
    plt.plot(x,y_train[:,0],'b',label='target_x')
    plt.scatter(x,prediction[:,0], label='pred_x')
    plt.plot(x,y_train[:,1],'r',label='target_y')
    plt.scatter(x,prediction[:,1], label='pred_y')
    plt.plot(x,y_train[:,2],'g',label='target_z')
    plt.scatter(x,prediction[:,2], label='pred_z')
    plt.xlabel('Time step')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper left')
    plt.show()

    ###TEST###
    y_test = []
    y_test_1 = xyz_target[training_step-input_size:training_step]
    v_1 = y_test_1
    v_1[:,0] = (v_1[:,0] - min_x) / (max_x - min_x) + 4
    v_1[:,1] = (v_1[:,1] - min_y) / (max_y - min_y) + 4
    v_1[:,2] = (v_1[:,2] - min_z) / (max_z - min_z) + 4

    for i in tqdm(range(test_step)):
        new_input_x_0 = time_series_process (0, v_1[:,0], t_relax).reshape(1, -1)
        new_input_x_1 = time_series_process (1, v_1[:,0], t_relax).reshape(1, -1)
        new_input_x_2 = time_series_process (2, v_1[:,0], t_relax).reshape(1, -1)

        new_input_y_0 = time_series_process (0, v_1[:,1], t_relax).reshape(1, -1)
        new_input_y_1 = time_series_process (1, v_1[:,1], t_relax).reshape(1, -1)
        new_input_y_2 = time_series_process (2, v_1[:,1], t_relax).reshape(1, -1)

        new_input_z_0 = time_series_process (0, v_1[:,2], t_relax).reshape(1, -1)
        new_input_z_1 = time_series_process (1, v_1[:,2], t_relax).reshape(1, -1)
        new_input_z_2 = time_series_process (2, v_1[:,2], t_relax).reshape(1, -1)

        new_input = np.hstack((new_input_x_0, new_input_x_1, new_input_x_2, new_input_y_0, new_input_y_1, new_input_y_2, new_input_z_0, new_input_z_1, new_input_z_2))
    
        prediction = mlr.predict(new_input)
        y_test.append(prediction)
        y_test_1 = np.vstack((y_test_1, prediction))
        y_test_1 = np.delete(y_test_1, 0, axis=0)
        if 0 <= i % update_interval <= update:
            y_test_1 = xyz_target[training_step-input_size+i:training_step+i]
            v_1 = y_test_1
        v_1[:,0] = (v_1[:,0] - min_x) / (max_x - min_x) + 4
        v_1[:,1] = (v_1[:,1] - min_y) / (max_y - min_y) + 4
        v_1[:,2] = (v_1[:,2] - min_z) / (max_z - min_z) + 4
    
    y_predict = mlr.predict(x_train)
    y_tested = y_predict
    for i in range(len(y_test)):
        y_tested = np.vstack((y_tested, y_test[i]))

    x = np.arange(input_size+1, training_step+test_step+1)

    lr_x_1, lr_y_1, lr_z_1 = mk_lorentz_attractor()
    xyz_target = np.vstack((lr_x_1, lr_y_1, lr_z_1)).T
    y_answer = xyz_target[input_size:training_step+test_step].reshape(-1, 3)
    x = x[-100-test_step:]
    plt.figure(dpi = 300, figsize=(10,3))
    plt.plot(x,y_answer[-100-test_step:,0],'b',label='target_x')
    plt.plot(x,y_tested[-100-test_step:,0], 'r',label='pred_x')
    plt.plot(x,y_answer[-100-test_step:,1],'b',label='target_y')
    plt.plot(x,y_tested[-100-test_step:,1], 'r',label='pred_y')
    plt.plot(x,y_answer[-100-test_step:,2],'b',label='target_z')
    plt.plot(x,y_tested[-100-test_step:,2], 'r',label='pred_z')
    plt.xlabel('Time step')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper left')

    # nrmse_train = calculate_rmse(y_answer[:training_step-input_size], y_tested[:training_step-input_size])
    # nrmse = calculate_rmse_last_k(y_answer, y_tested, test_step)
    # print(f'NRMSE for {training_step} training step: {nrmse_train}')
    # print(f'NRMSE for {test_step} test step: {nrmse}')

    return y_tested
