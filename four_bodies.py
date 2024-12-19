import numpy as np
import pandas as pd
import random
from numpy import infty
from tqdm import tqdm
from capytaine.ui.vtk import Animation
import xarray as xr
import matplotlib
from numpy import reshape
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D
import capytaine as cpt
from capytaine.io.xarray import separate_complex_values
from capytaine.io.legacy import export_hydrostatics 
from   capytaine.io.xarray import merge_complex_values
from capytaine.post_pro import rao
from capytaine.meshes.symmetric import build_regular_array_of_meshes
from   capytaine.bem.airy_waves import airy_waves_free_surface_elevation
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import math
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import time
import logging
logging.basicConfig(level=logging.INFO)

# Record start time
start_time=time.time()


# Flow control
ML_analysis=False
single_analysis=True
# ML_analysis = True
# single_analysis = False
# optimisation = True
optimisation = False

# Import mesh file
mesh=cpt.load_mesh("D:\Rhino\cylinder.STL", file_format="stl",name="cylinder")
# radius=2, height=10

# define function for calculation of rao 
def calculate_rao(dataset, B_PTO=29000):
    # Extract variables from the dataset
    added_mass = dataset.added_mass.data
    radiation_damping = dataset.radiation_damping.data
    omega = dataset.omega.data
    period=dataset.period.data
    M = dataset.inertia_matrix.data
    hydrostatic_stiff = dataset.hydrostatic_stiffness.data
    F_diff = dataset.diffraction_force.data
    F_froude = dataset.Froude_Krylov_force.data

    # Get dimensions
    num_omega = len(omega)
    num_wave_direction = len(dataset.wave_direction)
    num_radiating_dof = len(dataset.radiating_dof)
    num_influenced_dof = len(dataset.influenced_dof)

    # Initialize arrays
    B_pto = B_PTO  # Ns/m
    M_total = np.zeros((num_omega, num_radiating_dof, num_influenced_dof), dtype=np.complex128)
    C = np.zeros((num_omega, num_radiating_dof, num_influenced_dof), dtype=np.complex128)
    C_inverse = np.zeros((num_omega, num_radiating_dof, num_influenced_dof), dtype=np.complex128)
    RAO = np.zeros((num_omega, num_wave_direction, num_influenced_dof), dtype=np.complex128)

    
    for i in range(num_omega):
       
        # Merge mass matrix and added mass matrix
        M_total[i, :, :] = M + added_mass[i, :, :]

        # Setup C_ij
        C[i, :, :] = -(omega[i])**2 * M_total[i, :, :] + 1j * omega[i] * (radiation_damping[i, :, :] + B_pto) + hydrostatic_stiff[:, :]
        C_inverse[i, :, :] = np.linalg.inv(C[i, :, :])
    

    # Merge excitation force
    F_excitation = F_froude + F_diff

    # Calculate RAO
    for i in range(num_omega):
        for j in range(num_wave_direction):
            RAO[i,j,:] = np.dot(C_inverse[i, :, :], F_excitation[i,j,:])
            
    # Convert RAO to xarray.DataArray
    RAO_xarray = xr.DataArray(
        RAO,
        dims=['omega', 'wave_direction', 'influenced_dof'],
        coords={
            'omega': omega,
            'period': (('omega',), period),
            'wave_direction': dataset.wave_direction.data,
            'influenced_dof': dataset.influenced_dof.data
            
        },
        name='RAO'
    )
    return RAO_xarray

# set up default significant wave height range
min_wave = 0.25
max_wave = 16.75
default_wave_height_range=np.arange(min_wave, max_wave, 0.25)

# define function for calculation of power
def calculate_power(RAO, significant_wave_height=default_wave_height_range, B_pto=29000):
   # Extract omega from RAO DataArray
    omega=RAO.coords['omega'].data
    period=RAO.coords['period'].data
    influenced_dof=RAO.coords['influenced_dof'].data

    # Convert RAO DataArray to np.array
    RAO_array = RAO.data
    
    # Initialize variables
    num_heights = len(significant_wave_height)
    num_omega = len(omega)
    num_device = len(influenced_dof)
    num_wave_direction = RAO.shape[1]
    
    
    z_a = np.zeros((num_omega, num_heights, num_device, num_wave_direction))
    z_sign = significant_wave_height/2  # significant amplitude
    power = np.zeros((num_omega, num_heights, num_device, num_wave_direction))

    
    # Calculate response heave motion
    for i in range(num_omega):
        for j in range(num_heights):
            for k in range(num_device):
                for l in range(num_wave_direction):
                    z_a[i, j, k, l] = z_sign[j] * np.abs(RAO_array[i, l, k])

    # Calculate power
    for i in range(num_omega):
        for j in range(num_heights):
            for k in range(num_device):
                for l in range(num_wave_direction):
                    power[i, j, k, l] = 0.5 * (omega[i])**2 * B_pto * (z_a[i, j, k, l])**2 / 1000  # kW

    power_total = np.sum(power, axis=2) # size of power_total(i, j, l) represent(num_omega, num_wave_height, num_wave_direction)
    
    # Converge power to xarray.DataArray
    power_total_xarray = xr.DataArray(
        power_total,
        dims=['omega', 'wave_height', 'wave_direction'],
        coords={
            'omega': omega,
            'period': (('omega',), period),
            'wave_height': significant_wave_height,
            'wave_direction': RAO.coords['wave_direction'].data
            
        },
        name='power_total'
    )

    return power_total_xarray


period_default = np.arange(4.25, 14.75, 0.1)
def calculate_power_with_default_period(body, significant_wave_height, period_range=period_default, direction_range=[0]):
    radiation_problems_one_buoy = [cpt.RadiationProblem(body=body, radiating_dof=dof, period=period)
                                    for dof in body.dofs
                                    for period in period_range]
            

    diffraction_problems_one_buoy = [cpt.DiffractionProblem(body=body, wave_direction=direction, period=period)
                                    for period in period_range
                                    for direction in direction_range]

    solver = cpt.BEMSolver()
    radiation_results_one_buoy = solver.solve_all(radiation_problems_one_buoy)
    diffraction_results_one_buoy = solver.solve_all(diffraction_problems_one_buoy)
    dataset_one_buoy = cpt.assemble_dataset(radiation_results_one_buoy+diffraction_results_one_buoy)
    RAO_one_buoy = calculate_rao(dataset_one_buoy)
    power_one_buoy = calculate_power(RAO_one_buoy, significant_wave_height=significant_wave_height)
    return power_one_buoy
# sys.exit()    

def generate_array(array_design):
    mesh=cpt.load_mesh("D:\Rhino\cylinder.STL", file_format="stl",name="cylinder")
    body = cpt.FloatingBody(mesh=mesh, name="cylinder")
    body.center_of_mass = body.rotation_center = np.array([0,0,0])
    body.add_translation_dof(name='Heave')
    body.inertia_matrix = body.compute_rigid_body_inertia() 
    body.hydrostatic_stiffness = body.immersed_part().compute_hydrostatic_stiffness()
    all_buoys=body.assemble_arbitrary_array(array_design)

    return all_buoys

def setup_animation(body, fs, omega, wave_amplitude, wave_direction):
    # SOLVE BEM PROBLEMS
    bem_solver = cpt.BEMSolver()
    radiation_problems = [cpt.RadiationProblem(omega=omega, body=body.immersed_part(), radiating_dof=dof) for dof in body.dofs]
    radiation_results = bem_solver.solve_all(radiation_problems)
    diffraction_problem = cpt.DiffractionProblem(omega=omega, body=body.immersed_part(), wave_direction=wave_direction)
    diffraction_result = bem_solver.solve(diffraction_problem)

    dataset = cpt.assemble_dataset(radiation_results + [diffraction_result])
    rao_result = rao(dataset, wave_direction=wave_direction)

    # COMPUTE FREE SURFACE ELEVATION
    # Compute the diffracted wave pattern
    diffraction_elevation = bem_solver.get_free_surface_elevation(diffraction_result, fs)
    incoming_waves_elevation = fs.incoming_waves(diffraction_result)

    # Compute the wave pattern radiated by the RAO
    radiation_elevations_per_dof = {res.radiating_dof: (-1j*omega)*bem_solver.get_free_surface_elevation(res, fs) for res in radiation_results}
    radiation_elevation = sum(rao_result.sel(omega=omega, radiating_dof=dof).data * radiation_elevations_per_dof[dof] for dof in body.dofs)

    # SET UP ANIMATION
    # Compute the motion of each face of the mesh for the animation
    rao_faces_motion = sum(rao_result.sel(omega=omega, radiating_dof=dof).data * body.dofs[dof] for dof in body.dofs)

    # Set up scene
    animation = Animation(loop_duration=2*np.pi/omega)
    animation.add_body(body, faces_motion=wave_amplitude*rao_faces_motion)
    animation.add_free_surface(fs, wave_amplitude * (incoming_waves_elevation + diffraction_elevation + radiation_elevation))
    return animation

# Set up body
cylinder1=cpt.FloatingBody(mesh=mesh,name='body1')
cylinder1.center_of_mass=np.array([0,0,0])
cylinder1.rotation_center=np.array([0,0,0])
cylinder1.add_translation_dof(name="Heave")
cylinder1.keep_immersed_part()
cylinder1.inertia_matrix = cylinder1.compute_rigid_body_inertia()
cylinder1.hydrostatic_stiffness = cylinder1.compute_hydrostatic_stiffness(rho=1000)

# def create_square_array(size):
#     return np.array([[0, 0], [0, size], [size, 0], [size, size]])

def create_square_array(size):
    return np.array([[-size/2, -size/2], [-size/2, size/2], [size/2, -size/2], [size/2, size/2]], dtype=np.int32)


# Interval spacing distance for different square arrays (unit: meter)
sizes = [8, 16, 24, 32, 40, 48]

# List to store the arrays
array_designs = []

# Loop to create and store the arrays
for size in sizes:
    array_designs.append(create_square_array(size))

all_buoys_list = []

# test array
test_array=np.array([[-20,-20],[-20,20],[20,-20],[20,20]])
test_buoys=cylinder1.assemble_arbitrary_array(test_array)
# test_buoys.show()

# Loop to apply each array design to the buoys
for array_design in array_designs:
    all_buoys = cylinder1.assemble_arbitrary_array(array_design)
    all_buoys_list.append(all_buoys)

# Check the array shape 
# all_buoys_list[5].show()

# Create 4 cylinders in different array
array_design_square=np.array([[0,0],[0,40],[40,0],[40,40]])
# array_design_line=np.array([[0,60],[0,20],[0,-20],[0,-60]])
# array_design_align_line=np.array([[60,0],[20,0],[-20,0],[-60,0]])
# array_design_diamond=np.array([[0,0],[0,40],[34,20],[-34,20]],dtype=int)
# array_design_parallelogram=np.array([[0,0],[20,34],[40,0],[60,34]],dtype=int)
# array_design_triangle=np.array([[0,30],[40,30],[20,18.45],[20,-4.64]])
# array_design_gun=np.array([[0,0],[0,40],[40,40],[80,40]])

# Set up WEC farm with different array
# all_buoys=cylinder1.assemble_arbitrary_array(array_design_square)
# all_buoys.show()


# Make the other three device fixed to plot the radiation field in single analysis
case1_bodies=cylinder1.assemble_regular_array(40,(2,2))
case1_bodies.keep_only_dofs(dofs=['0_0__Heave'])
case2_bodies=cylinder1.assemble_regular_array(40,(2,2))
case2_bodies.keep_only_dofs(dofs=['0_1__Heave'])
case3_bodies=cylinder1.assemble_regular_array(40,(2,2))
case3_bodies.keep_only_dofs(dofs=['1_0__Heave'])
case4_bodies=cylinder1.assemble_regular_array(40,(2,2))
case4_bodies.keep_only_dofs(dofs=['1_1__Heave'])
max_period=16.75
min_period=0.25


if ML_analysis:
    # omega_range=np.linspace(0.5,2,16) # activate when exporting ncfile to do regression analysis in matlab
    omega_range=np.linspace(0.01,10,100)
    period_range=np.arange(min_period, max_period, 0.25)
    omega_range_p=2*math.pi/period_range
    direction_range=[0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi, 5*math.pi/2, 3*math.pi/2, 7*math.pi/4]
    F_excitation_all = []
    RAO_data=[]
    RAO_data_in_period=[]
    power_data=[]
    power_data_in_period=[]

       
    # Loop to perform hydrodynamic analysis for every array design
    for i, all_buoys in enumerate(all_buoys_list, start=1):
        radiation_problems = [cpt.RadiationProblem(body=all_buoys, radiating_dof=dof, omega=omega)
                    for dof in all_buoys.dofs
                    for omega in omega_range]
            

        diffraction_problems = [cpt.DiffractionProblem(body=all_buoys, wave_direction=direction, omega=omega)
                    for omega in omega_range
                    for direction in direction_range]
        
        radiation_problems_in_period = [cpt.RadiationProblem(body=all_buoys, radiating_dof=dof, period=period)
                    for dof in all_buoys.dofs
                    for period in period_range]
            

        diffraction_problems_in_period = [cpt.DiffractionProblem(body=all_buoys, wave_direction=direction, period=period)
                    for period in period_range
                    for direction in direction_range]
        
        # Solve the Capytaine problem
        solver = cpt.BEMSolver()
        radiation_results = solver.solve_all(radiation_problems)
        diffraction_results=solver.solve_all(diffraction_problems)
        dataset = cpt.assemble_dataset(radiation_results+diffraction_results)

        radiation_results_in_period = solver.solve_all(radiation_problems_in_period)
        diffraction_results_in_period = solver.solve_all(diffraction_problems_in_period)
        dataset_in_period = cpt.assemble_dataset(radiation_results_in_period+diffraction_results_in_period)
     

        # Export output data
        # filename = f"D:/capytaine_data/array_design_{i}.nc"
        # separate_complex_values(dataset).to_netcdf(filename,
        #             encoding={'radiating_dof': {'dtype': 'U'},
        #                         'influenced_dof': {'dtype': 'U'}})

        F_combined = np.real(dataset.diffraction_force) + np.real(dataset.Froude_Krylov_force)
        F_excitation = np.abs(F_combined)
        
        # Append the combined excitation force values to the list
        F_excitation_all.append(F_excitation.data)

        # Append rao result
        RAO_in_omega=calculate_rao(dataset)
        RAO_data.append(RAO_in_omega)
        RAO_in_period=calculate_rao(dataset_in_period)
        RAO_data_in_period.append(RAO_in_period)
        
        # Append power result
        power=calculate_power(RAO_in_omega)
        power_data.append(power)
        power_in_period=calculate_power(RAO_in_period)
        power_data_in_period.append(power_in_period)



    
    # Extract data
    sign_wave_height_range=power_in_period.coords['wave_height'].data
    
    # Verify power
    # power_array_40=power_data_in_period[4]
    # print("power for spacing=40 is:", power_array_40[23,35,0])

    
    # Verify RAO plot with plot generated from Matlab
    # plt.figure()
    # plt.title("RAO device 1")
    # RAO_data=RAO_pto[4] # number represent the index of array_design
    # plt.plot(omega_range,np.abs(RAO_data[:,0,0])) # RAO_data[omega, wave_direction, device_index]
    # plt.xlabel("omega(rad/s)")
    # plt.ylabel("RAO(-)")
    # plt.show()
    # print("RAO shape in second coordinate",RAO_data.shape[1])

    # Predefined random states for each loop(can be any value)
    random_states = [7, 178, 101, 303]

    # Prepare data for machine learning for each device separately
    for device_index in range(4):
        data = []
        

        for array_idx, F_excitation in enumerate(F_excitation_all):
            for i in range(F_excitation.shape[0]):  # Frequency dimension
                for j in range(F_excitation.shape[1]):  # Wave angle dimension
                    data.append([omega_range[i], direction_range[j], sizes[array_idx], F_excitation[i, j, device_index]])

       
        df = pd.DataFrame(data, columns=['Frequency', 'Wave_Angle', 'Array_Design', 'Load'])
        
        # Splitting the data for load
        X = df[['Frequency', 'Wave_Angle', 'Array_Design']]
        y = df['Load']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Load analysis
        # Standardizing the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Save the scaler
        joblib.dump(scaler, f'scaler_load_device_{device_index + 1}.joblib')

        print(f'Training Random Forest for load on device {device_index + 1}...')
        # Random Forest Regressor for load
        random_state_rf = random_states[device_index]
        rf = RandomForestRegressor(n_estimators=10000, random_state=random_state_rf)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)

        # Save the Random Forest model
        joblib.dump(rf, f'rf_load_device_{device_index + 1}.joblib')

        # Evaluate the model
        mse_rf = mean_squared_error(y_test, y_pred_rf)
        r2_rf = r2_score(y_test, y_pred_rf)
        print(f"Load - Random Forest - MSE: {mse_rf}, R^2: {r2_rf}")

        print(f'Training Neural Network for load on device {device_index + 1}...')
        # Neural Network Regressor for load
        random_state_mlp = random_states[device_index]
        mlp = MLPRegressor(hidden_layer_sizes=(100,), max_iter=50000, random_state=random_state_mlp)
        mlp.fit(X_train, y_train)
        y_pred_mlp = mlp.predict(X_test)

        # Save the Neural Network model
        joblib.dump(mlp, f'mlp_load_device_{device_index + 1}.joblib')

        # Evaluate the model
        mse_mlp = mean_squared_error(y_test, y_pred_mlp)
        r2_mlp = r2_score(y_test, y_pred_mlp)
        print(f"Load - Neural Network - MSE: {mse_mlp}, R^2: {r2_mlp}")

        # Adding an end time and print statement
        end_load_time = time.time()
        end_load_elapsed_time = end_load_time - start_time
        minutes, seconds = divmod(end_load_elapsed_time, 60)
        print(f"Device {device_index + 1} training completed in {int(minutes)} minutes and {int(seconds)} seconds")

        # Plotting the actual vs predicted loads for each device
        plt.figure(device_index + 1, figsize=(12, 6))
        plt.suptitle(f'Device {device_index + 1} Load Machine Learning Analysis')

        plt.subplot(1, 2, 1)
        plt.scatter(y_test, y_pred_rf, color='blue', alpha=0.5)
        plt.xlabel('Actual Loads(N)')
        plt.ylabel('Predicted Loads(N)')
        plt.title('Random Forest Regressor')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        plt.text(0.05, 0.95, f'R² = {r2_rf:.2f}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')

        plt.subplot(1, 2, 2)
        plt.scatter(y_test, y_pred_mlp, color='red', alpha=0.5)
        plt.xlabel('Actual Loads(N)')
        plt.ylabel('Predicted Loads(N)')
        plt.title('Neural Network Regressor')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        plt.text(0.05, 0.95, f'R² = {r2_mlp:.2f}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show(block=False)


    # Power analysis
    data_power = []
    for array_idx, power_in_period in enumerate(power_data_in_period):
                for i in range(power_in_period.shape[0]):  # Frequency dimension
                    for j in range(power_in_period.shape[1]):  # Significant wave height dimension
                        for k in range(power_in_period.shape[2]):  # Wave angle dimension
                            data_power.append([omega_range_p[i], sign_wave_height_range[j], direction_range[k], sizes[array_idx], power_in_period[i, j, k]])
    
    df_p = pd.DataFrame(data_power, columns=['Frequency', 'Significant_Wave_Height', 'Wave_Angle', 'Array_Design', 'Power'])
    
    # Splitting the data for power
    X_p = df_p[['Frequency', 'Significant_Wave_Height', 'Wave_Angle', 'Array_Design']]
    y_p = df_p['Power']
    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_p, y_p, test_size=0.2, random_state=42)

    
    # Standardizing the features
    scaler_p = StandardScaler()
    X_train_p = scaler_p.fit_transform(X_train_p)
    X_test_p = scaler_p.transform(X_test_p)

    # Save the scaler
    joblib.dump(scaler_p, 'scaler_power_device.joblib')
    
    print('Training Random Forest for power...')
    # Random Forest Regressor for power
    rf_p = RandomForestRegressor(n_estimators=100, random_state=303)
    rf_p.fit(X_train_p, y_train_p)
    y_pred_rf_p = rf_p.predict(X_test_p)

    # Save the Random Forest model
    joblib.dump(rf_p, f'rf_power_device.joblib')

    # Evaluate the model
    mse_rf_p = mean_squared_error(y_test_p, y_pred_rf_p)
    r2_rf_p = r2_score(y_test_p, y_pred_rf_p)
    print(f'Power - Random Forest - MSE: {mse_rf_p}, R^2: {r2_rf_p}')

    # Adding an end time and print statement
    end_rf_power_time = time.time()
    end_rf_power_elapsed_time = end_rf_power_time - start_time
    minutes, seconds = divmod(end_rf_power_elapsed_time, 60)
    print(f"Random Forest training for power completed in {int(minutes)} minutes and {int(seconds)} seconds")
    # Here comment
    print('Training Neural Network for power...')
    # Neural Network Regressor for power (NOT RELIABLE)
    mlp_p = MLPRegressor(hidden_layer_sizes=(100,), max_iter=100000, random_state=303)
    mlp_p.fit(X_train_p, y_train_p)
    y_pred_mlp_p = mlp_p.predict(X_test_p)

    # Save the Neural Network model
    joblib.dump(mlp_p, f'mlp_power_device_{device_index + 1}.joblib')

    # Evaluate the model
    mse_mlp_p = mean_squared_error(y_test_p, y_pred_mlp_p)
    r2_mlp_p = r2_score(y_test_p, y_pred_mlp_p)
    print(f'Power - Neural Network - MSE: {mse_mlp_p}, R^2: {r2_mlp_p}')
    # Here end comment
    # Plotting the actual vs predicted power for each device
    plt.figure(5, figsize=(12, 6))
    plt.suptitle('Power Machine Learning Analysis')

    plt.subplot(1, 2, 1) # HERE COMMENT
    plt.scatter(y_test_p, y_pred_rf_p, color='blue', alpha=0.5)
    plt.xlabel('Actual Power(kW)')
    plt.ylabel('Predicted Power(kW)')
    plt.title('Random Forest Regressor')
    plt.plot([y_test_p.min(), y_test_p.max()], [y_test_p.min(), y_test_p.max()], 'k--', lw=2)
    plt.text(0.05, 0.95, f'R² = {r2_rf_p:.2f}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')

   
    plt.subplot(1, 2, 2)
    plt.scatter(y_test_p, y_pred_mlp_p, color='red', alpha=0.5)
    plt.xlabel('Actual Power(kW)')
    plt.ylabel('Predicted Power(kW)')
    plt.title('Neural Network Regressor')
    plt.plot([y_test_p.min(), y_test_p.max()], [y_test_p.min(), y_test_p.max()], 'k--', lw=2)
    plt.text(0.05, 0.95, f'R² = {r2_mlp_p:.2f}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show(block=False)    


    
    end_time=time.time()
    total_elapsed_time=end_time - start_time
    minutes, seconds = divmod(total_elapsed_time, 60)
    print(f'Total execution time: {int(minutes)} minutes and {int(seconds)} seconds')
    plt.show()








if single_analysis:
    
    all_buoys = cylinder1.assemble_arbitrary_array(array_design_square)
    
    omega_range = 2
    wave_direction = 0

    # Body control (single body only for plotting radiation field)
    # body = all_buoys
    body = case1_bodies
    # body = case2_bodies
    # body = case3_bodies
    # body = case4_bodies

    radiation_problems = [cpt.RadiationProblem(omega=omega_range, body=body, water_depth=infty,radiating_dof=dof) for dof in body.dofs]
    
    diffraction_problem = cpt.DiffractionProblem(omega=omega_range, wave_direction=wave_direction, body=body, water_depth=infty)

    # Solve the Capytaine problem
    solver = cpt.BEMSolver()
    radiation_result = solver.solve_all(radiation_problems)
    diffraction_result=solver.solve(diffraction_problem)

   
    
    # Plot pressure field
    combined_radiation_pressure = np.zeros_like(np.real(radiation_result[0].pressure))
    for i in range(len(radiation_result)):
        combined_radiation_pressure += np.real(radiation_result[i].pressure)
    
    body.show_matplotlib(
        color_field=np.real(combined_radiation_pressure+diffraction_result.pressure),
        cmap=plt.get_cmap("viridis"),  # Colormap
        cbar_label="Pressure(Pa)"
        )
    

    #Postprocessing
    dataset = cpt.assemble_dataset(radiation_result+[diffraction_result])
    rao_result = rao(dataset,wave_direction=wave_direction)
    rao_result_manual=calculate_rao(dataset, B_PTO=0)

    # Set up animation
    all_buoys_animation=generate_array(array_design_square)
    fs = cpt.FreeSurface(x_range=(-100, 75), y_range=(-100, 75), nx=100, ny=100)
    anim = setup_animation(all_buoys_animation, fs, omega=omega_range, wave_amplitude=1, wave_direction=wave_direction)
    anim.run(camera_position=(-60, -60, 50), resolution=(800, 600))
    anim.save(f"animated_array_freq={omega_range}.ogv", camera_position=(-60, -60, 50), resolution=(800, 600))

    # Verify rao generated from manual calculation wtih rao generated from default function in Capytaine
    # print("RAO in the 4 dof are", rao_result)
    # print('RAO from manual calculation in the 4 dof:', rao_result_manual )
    
    print('Calculating wave field...')
   # Set up free surface mesh
    xmin=-32
    xmax=68
    ymin=-32
    ymax=68
    nx=100
    ny=100
    x  = np.linspace(xmin, xmax, nx, endpoint=True)
    y  = np.linspace(ymin, ymax, ny, endpoint=True)
    points = np.meshgrid(x, y, np.linspace(-100.0, 0.0, 100))
    grid=np.meshgrid(x,y)
    fs = cpt.FreeSurface(x_range=(xmin, xmax), y_range=(ymin, ymax), nx=nx, ny=ny, name = 'free_surface')

  
    # Compute individual wave patterns:
    incoming_waves_elevation = airy_waves_free_surface_elevation(grid,diffraction_result) 
    diffraction_elevation = solver.compute_free_surface_elevation(grid,diffraction_result)
    radiation_elevation_per_dof = {rad.radiating_dof: solver.compute_free_surface_elevation(grid, rad) for rad in radiation_result}
    radiation_elevation=sum(rao_result.sel(radiating_dof=dof).data * radiation_elevation_per_dof[dof] for dof in body.dofs)
    # potential=solver.compute_potential(points, diffraction_result) 
    wave_field= incoming_waves_elevation + diffraction_elevation + radiation_elevation
    
    
    # circle_coords = test_array
    circle_coords = array_design_square
    circle_numbers = [1, 2, 3, 4]
    
    def plot_with_circles(grid, data, title):
        plt.figure()
        plt.pcolormesh(grid[0], grid[1], np.real(data))
        plt.xlabel("x")
        plt.ylabel("y")
        cbar = plt.colorbar()
        cbar.set_label('Elevation (m)')
        plt.title(title)
    
        # Add white-filled circles
        for (x, y), num in zip(circle_coords, circle_numbers):
            plt.scatter(x, y, color='white', edgecolor='black', s=200, zorder=5)
            plt.text(x, y, str(num), color='black', fontsize=12, ha='center', va='center', zorder=6)


   

   # Plot each of the figures with circles
    plot_with_circles(grid, incoming_waves_elevation, 'Incoming waves elevation')
    plot_with_circles(grid, diffraction_elevation, 'Diffraction elevation')
    plot_with_circles(grid, radiation_elevation, 'Radiation elevation')
    plot_with_circles(grid, wave_field, 'Total wave field')


    plt.show()


# Test the machine learning model
if optimisation:
    # specific_sea = True
    totally_random = False
    specific_sea = False
    # totally_random = True
    # random_sea = False
    random_sea = True

    if random_sea:
        np.random.seed(100)
        print('Generating random values...')
        num_wave_direction_samples = 100
        num_array_samples = 100
        wave_direction_random_values = np.random.uniform(0, 2*np.pi, num_wave_direction_samples)
        wave_direction_in_order = np.sort(wave_direction_random_values)
        array_random_spacing = np.random.randint(4, 49, num_array_samples)
        array_in_order = np.sort(array_random_spacing)
        # Data used for wave scatter diagram
        selected_period_range = np.arange(4.25, 14.75, 0.5)
        omega_values = 2*np.pi/selected_period_range
        wave_height_values = np.arange(14.25, -0.25, -0.5)

        num_wave_height = len(wave_height_values)
        num_omega = len(omega_values)
        

        # couple all data (wave frequency, wave height, array spacing, wave direction)
        test=[]
        for i in range(num_wave_height):
            for j in range(num_omega):
                for k in range(num_wave_direction_samples):
                    for m in range(num_array_samples):
                        test.append([wave_height_values[i], omega_values[j], wave_direction_in_order[k], array_in_order[m]])
        
        test_array = np.array(test)
        omega_selected_values = test_array[:, 1]
        wave_height_selected_values = test_array[:, 0]
        array_spacing = test_array[:, 3]
        wave_direction_selected_values = test_array[:, 2]
        
        # Manually input wave scatter diagram (unit=[%])
        occurence_diagram = np.array([
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01, 0.02, 0.01, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02, 0.03, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.03, 0.01, 0, 0, 0.02],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01, 0.02, 0, 0.01, 0.01, 0.02],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01, 0.02, 0, 0.03, 0.01, 0.01, 0.01],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01, 0, 0, 0.02, 0.03, 0.05, 0.01, 0.01, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02, 0.02, 0.05, 0.05, 0.02, 0, 0.01],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02, 0.04, 0.09, 0.09, 0.06, 0.02, 0.01, 0.01],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01, 0, 0.04, 0.09, 0.13, 0.11, 0.09, 0.01, 0.02, 0.02, 0.01],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02, 0.02, 0.07, 0.14, 0.14, 0.12, 0.03, 0.02, 0.02, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02, 0.07, 0.14, 0.24, 0.21, 0.14, 0.11, 0.03, 0.02, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0.06, 0.11, 0.28, 0.34, 0.35, 0.31, 0.22, 0.14, 0.05, 0.02, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0.01, 0.08, 0.12, 0.52, 0.6, 0.72, 0.47, 0.27, 0.2, 0.12, 0.08, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0.04, 0.16, 0.5, 0.83, 0.95, 0.86, 0.55, 0.52, 0.37, 0.1, 0.03, 0, 0.05, 0.01, 0.01],
                                    [0, 0, 0, 0, 0, 0.07, 0.11, 0.51, 0.89, 1.49, 2.06, 1.57, 1.05, 0.68, 0.51, 0.34, 0.1, 0.08, 0.05, 0, 0],
                                    [0, 0, 0, 0, 0.09, 0.21, 0.74, 1.15, 2.14, 2.61, 2.58, 1.58, 1.15, 0.9, 0.56, 0.39, 0.26, 0.09, 0.1, 0.02, 0.01],
                                    [0, 0, 0.01, 0.02, 0.17, 0.57, 1.71, 2.03, 2.15, 2.42, 1.87, 1.53, 1.18, 0.68, 0.36, 0.21, 0.14, 0.15, 0.13, 0.09, 0.03],
                                    [0, 0, 0.11, 0.67, 1.08, 1.74, 1.93, 2.81, 3.43, 3.71, 2.68, 1.79, 1.15, 0.49, 0.2, 0.17, 0.1, 0.03, 0.02, 0, 0.01],
                                    [0, 0.02, 0.46, 1.34, 0.92, 1.5, 2.18, 2.38, 3.1, 2.24, 2.28, 1.74, 1.02, 0.55, 0.2, 0.03, 0.05, 0.05, 0, 0, 0],
                                    [0, 0.09, 0.14, 0.55, 0.3, 0.71, 0.95, 0.79, 0.49, 0.33, 0.4, 0.14, 0.06, 0.03, 0.02, 0, 0.01, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                                ])
        
        # transfer percentage to decimal
        data=occurence_diagram/100 

        # Create a DataFrame for the machine learning input
        input_data = {
            'Frequency': omega_selected_values,
            'Significant_Wave_Height': wave_height_selected_values,
            'Wave_Angle': wave_direction_selected_values,
            'Array_Design': array_spacing
        }
        input_df = pd.DataFrame(input_data)

        # Load the scaler and models
        scaler = joblib.load('scaler_power_device.joblib')
        rf = joblib.load('rf_power_device.joblib')

        # Standardize the input data
        input_scaled = scaler.transform(input_df)

        # Make predictions using the trained model
        predicted_power = rf.predict(input_scaled)

        # Add the predicted power to the DataFrame
        input_df['Predicted_Power'] = predicted_power

        # Display the DataFrame
        # print(input_df)
        # print(input_df.iloc[0:27]) #verify data

        # Reshape the predicted power to the same dimension of the wave scatter diagram
        data_power = predicted_power.reshape((num_wave_height, num_omega, num_wave_direction_samples, num_array_samples))
        
        # Calculate the most probable power output
        total_power_list=[]
        for i in range(num_wave_direction_samples):
            for j in range(num_array_samples):
                power_scatter = data_power[:, :, i, j]*data
                total_power = np.sum(power_scatter)
                wave_direction_value = wave_direction_in_order[i]
                array_value = array_in_order[j]
                total_power_list.append([total_power, wave_direction_value, array_value])
        
        total_power_array = np.array(total_power_list)
        maximum_power = total_power_array.max(axis=0)
        print(maximum_power[0]) 
        
        print(f'Most probable power output in the given site is:{int(maximum_power[0])} kW')
        print(f'With array spacing is: {int(maximum_power[2])} m and wave direction angle is: {maximum_power[1]:.2f}')

        # Prepare data to calculate q factor
        # Calculate power for one device
        one_buoy_power = calculate_power_with_default_period(body=cylinder1, significant_wave_height=wave_height_values)
        omega_indices = list(range(0, 105, 5)) # select power value at period 4.25, 4.75, 5.25.....
        subset = one_buoy_power.isel(omega=omega_indices, wave_height=slice(None), wave_direction=slice(None))
        one_buoy_power_data = subset.data
        one_buoy_power_array = one_buoy_power_data.transpose(1, 0, 2) #Alligned with the shape of wave scatter data
    
        total_power_one_buoy_list=[]
        for i in range(one_buoy_power_array.shape[2]):
            power_one_buoy_scatter = one_buoy_power_array[:, :, i]*data
            total_power_one_buoy = np.sum(power_one_buoy_scatter)
            total_power_one_buoy_list.append(total_power_one_buoy)

        total_power_one_buoy_array = np.array(total_power_one_buoy_list)
        q_factor = total_power_array[:, 0]/(4*total_power_one_buoy_array[0])

        # Store data in DataFrame
        q_factor_collection = {
            'Wave_Direction [rad]': total_power_array[:, 1],
            'Array_Spacing [m]': total_power_array[:, 2],
            'Total_Power [kW]':total_power_array[:, 0],
            'q_factor [-]': q_factor
                }
            
        q_summary = pd.DataFrame(q_factor_collection)        
        # print(q_summary)

        # Find the index of the sample with the maximum predicted power
        max_q_factor_index = q_summary['q_factor [-]'].idxmax()

        # Get the sample with the maximum predicted power
        max_q_factor = q_summary.iloc[max_q_factor_index]

        # Show the design which result in the largest q factor
        print(max_q_factor)

        # Calculate execution time
        end_time=time.time()
        total_elapsed_time=end_time - start_time
        minutes, seconds = divmod(total_elapsed_time, 60)
        print(f'Total execution time: {int(minutes)} minutes and {int(seconds)} seconds')




    if specific_sea:
        ## Test the total output in a specific site when wave direction = 0, array spacing = 40 meters
        # Data used for wave scatter diagram
        selected_period_range = np.arange(4.25, 14.75, 0.5)
        omega_values = 2*np.pi/selected_period_range
        wave_height_values = np.arange(14.25, -0.25, -0.5)

        num_wave_height = len(wave_height_values)
        num_omega = len(omega_values)
        num_samples = num_wave_height*num_omega 
        
    
        # Couple omega data and wave height data
        test=[]
        for i in range(num_wave_height):
            for j in range(num_omega):
                test.append([wave_height_values[i], omega_values[j]])
        
        test_array = np.array(test)
        omega_selected_values = test_array[:, 1]
        wave_height_selected_values = test_array[:, 0]
        array_spacing = np.full(num_samples, 40) # Create 1D array with identical value
        wave_direction_selected_values = np.full(num_samples, 0)

        
        # Manually input wave scatter diagram (unit=[%])
        occurence_diagram = np.array([
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01, 0.02, 0.01, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02, 0.03, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.03, 0.01, 0, 0, 0.02],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01, 0.02, 0, 0.01, 0.01, 0.02],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01, 0.02, 0, 0.03, 0.01, 0.01, 0.01],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01, 0, 0, 0.02, 0.03, 0.05, 0.01, 0.01, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02, 0.02, 0.05, 0.05, 0.02, 0, 0.01],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02, 0.04, 0.09, 0.09, 0.06, 0.02, 0.01, 0.01],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01, 0, 0.04, 0.09, 0.13, 0.11, 0.09, 0.01, 0.02, 0.02, 0.01],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02, 0.02, 0.07, 0.14, 0.14, 0.12, 0.03, 0.02, 0.02, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02, 0.07, 0.14, 0.24, 0.21, 0.14, 0.11, 0.03, 0.02, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0.06, 0.11, 0.28, 0.34, 0.35, 0.31, 0.22, 0.14, 0.05, 0.02, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0.01, 0.08, 0.12, 0.52, 0.6, 0.72, 0.47, 0.27, 0.2, 0.12, 0.08, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0.04, 0.16, 0.5, 0.83, 0.95, 0.86, 0.55, 0.52, 0.37, 0.1, 0.03, 0, 0.05, 0.01, 0.01],
                                    [0, 0, 0, 0, 0, 0.07, 0.11, 0.51, 0.89, 1.49, 2.06, 1.57, 1.05, 0.68, 0.51, 0.34, 0.1, 0.08, 0.05, 0, 0],
                                    [0, 0, 0, 0, 0.09, 0.21, 0.74, 1.15, 2.14, 2.61, 2.58, 1.58, 1.15, 0.9, 0.56, 0.39, 0.26, 0.09, 0.1, 0.02, 0.01],
                                    [0, 0, 0.01, 0.02, 0.17, 0.57, 1.71, 2.03, 2.15, 2.42, 1.87, 1.53, 1.18, 0.68, 0.36, 0.21, 0.14, 0.15, 0.13, 0.09, 0.03],
                                    [0, 0, 0.11, 0.67, 1.08, 1.74, 1.93, 2.81, 3.43, 3.71, 2.68, 1.79, 1.15, 0.49, 0.2, 0.17, 0.1, 0.03, 0.02, 0, 0.01],
                                    [0, 0.02, 0.46, 1.34, 0.92, 1.5, 2.18, 2.38, 3.1, 2.24, 2.28, 1.74, 1.02, 0.55, 0.2, 0.03, 0.05, 0.05, 0, 0, 0],
                                    [0, 0.09, 0.14, 0.55, 0.3, 0.71, 0.95, 0.79, 0.49, 0.33, 0.4, 0.14, 0.06, 0.03, 0.02, 0, 0.01, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                                ])
        
        # transfer percentage to decimal
        data=occurence_diagram/100 

        # Create occurence dataframe
        # occurence_data_array = xr.DataArray(
        #     data,
        #     dims=['significant wave height', 'wave period'],
        #     coords={
        #         'wave period':selected_period_range,
        #         'significant wave height': wave_height_values
                
        #     },
        #     name='occurence scatter diagram'
        # )
        


        # Create a DataFrame for the machine learning input
        input_data = {
            'Frequency': omega_selected_values,
            'Significant_Wave_Height': wave_height_selected_values,
            'Wave_Angle': wave_direction_selected_values,
            'Array_Design': array_spacing
        }
        input_df = pd.DataFrame(input_data)

        # Load the scaler and models
        scaler = joblib.load('scaler_power_device.joblib')
        rf = joblib.load('rf_power_device.joblib')

        # Standardize the input data
        input_scaled = scaler.transform(input_df)

        # Make predictions using the trained model
        print('Predicting power based on Random Forest model...')
        predicted_power = rf.predict(input_scaled)

        # Add the predicted power to the DataFrame
        input_df['Predicted_Power'] = predicted_power

        # Display the DataFrame
        print(input_df)
        # Reshape the predicted power to the same dimension of the wave scatter diagram
        data_power = predicted_power.reshape((num_wave_height, num_omega))
        # Calculate the most probable power output
        total_power_array = data_power*data
        total_power = np.sum(total_power_array)
        print(f'Most probable power output in the given site is:{int(total_power)} kW')

    
        # Calculate execution time
        end_time=time.time()
        total_elapsed_time=end_time - start_time
        minutes, seconds = divmod(total_elapsed_time, 60)
        print(f'Total execution time: {int(minutes)} minutes and {int(seconds)} seconds')

    if totally_random:
        # Random input
        print('Generating random values...')
        np.random.seed(100)
        num_samples = 1000
        max_period = 16.75
        min_period = 0.5
        wave_height_random_values = np.random.choice(np.arange(min_wave, max_wave, 0.25), num_samples)
        max_omega = 2*np.pi/max_period
        min_omega = 2*np.pi/min_period
        omega_random_values = np.random.uniform(min_omega, max_omega, num_samples)
        wave_direction_random_values = np.random.uniform(0, 2*np.pi, num_samples)
        array_spacing = np.random.randint(4, 49, num_samples)

        # Create a DataFrame for the machine learning input
        input_data = {
            'Frequency': omega_random_values,
            'Significant_Wave_Height': wave_height_random_values,
            'Wave_Angle': wave_direction_random_values,
            'Array_Design': array_spacing
        }
        input_df = pd.DataFrame(input_data)

        # Load the scaler and models
        scaler = joblib.load('scaler_power_device.joblib')
        rf = joblib.load('rf_power_device.joblib')

        # Standardize the input data
        input_scaled = scaler.transform(input_df)

        # Make predictions using the trained model
        print('Predicting power based on Random Forest model...')
        predicted_power = rf.predict(input_scaled)

        # Add the predicted power to the DataFrame
        input_df['Predicted_Power'] = predicted_power

        # Display the DataFrame
        print(input_df)

        # Save the DataFrame to a CSV file
        # input_df.to_csv('D:/capytaine_data/predicted_power.csv')

        # Find the index of the sample with the maximum predicted power
        max_power_index = input_df['Predicted_Power'].idxmax()

        # Get the sample with the maximum predicted power
        max_power_sample = input_df.iloc[max_power_index]

        # Print the details of the sample with the maximum predicted power
        print("Sample with the maximum predicted power:")
        print(max_power_sample)

        # Calculate execution time
        end_time=time.time()
        total_elapsed_time=end_time - start_time
        minutes, seconds = divmod(total_elapsed_time, 60)
        print(f'Total execution time: {int(minutes)} minutes and {int(seconds)} seconds')
