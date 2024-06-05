from scipy.signal import find_peaks, butter, filtfilt
from scipy.stats import linregress
import pandas as pd
import numpy as np

# Low Pass Butterworth Filter
def low_pass_filter(data, cutoff=4, sample_rate=30.0, order=2):
    normal_cutoff = cutoff / (0.5 * sample_rate)
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)
    y = filtfilt(b, a, data)
    return y

def _add_velocity_and_speed(df: pd.DataFrame, sample_rate):
    df = df.sort_index(axis=1)
    for joint_name in set(df.columns.get_level_values(0)):  # für alle Gelenke Speed berechnen
        velocity = df[joint_name, 'Position'].diff()
        velocity = velocity.interpolate(axis=0, limit_direction='both')
        speed = np.linalg.norm(velocity[['X', 'Y', 'Z']].values, axis=1, ord=2)
        speed = low_pass_filter(speed, sample_rate=sample_rate)
        # speed here is cm/sample -> m/s
        speed = speed * sample_rate / 100

        df[joint_name, 'Velocity', 'X'] = velocity['X']
        df[joint_name, 'Velocity', 'Y'] = velocity['Y']
        df[joint_name, 'Velocity', 'Z'] = velocity['Z']
        df[joint_name, 'Speed', 'XYZ'] = speed
        df = df.sort_index(axis=1)
    return df

def _find_valley_indices(df, distance, height=None, threshold=None, prominence=0.2):
    return find_peaks(-(df).to_numpy().flatten(), height, threshold, distance, prominence)[0]

def calc_standing_foot_positions(df, distance=20, sample_rate=30.0):
    df = _add_velocity_and_speed(df, sample_rate)
    
    results = pd.DataFrame(columns=['i', 'pos', 'vec', 'stride_len'], index=['LFoot', 'LToe', 'RFoot', 'RToe'])
    for foot, toe in [('LFoot', 'LToe'), ('RFoot', 'RToe')]:
                
        valley_indices       = _find_valley_indices(df[foot, 'Speed'], distance=distance)
        
        results.at[foot, 'i']   = valley_indices
        results.at[foot, 'pos'] = df.iloc[valley_indices].loc[:, (foot, 'Position')]
        results.at[foot, 'vec'] = df.iloc[valley_indices][foot]['Position'].diff()
        results.at[foot, 'stride_len'] = np.linalg.norm(results['vec'][foot][['X', 'Y', 'Z']].values, axis=1, ord=2)[1:]
        
        results.at[toe, 'i']   = valley_indices
        results.at[toe, 'pos'] = df.iloc[valley_indices].loc[:, (toe, 'Position')]
        results.at[toe, 'vec'] = df.iloc[valley_indices][toe]['Position'].diff()
        results.at[toe, 'stride_len'] = np.linalg.norm(results['vec'][toe][['X', 'Y', 'Z']].values, axis=1, ord=2)[1:]   
    return results, df