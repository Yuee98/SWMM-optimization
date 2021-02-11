from pyswmm import Simulation, Nodes
import datetime
import re
import math
import numpy as np

# 读取真实数据
def readobservationfile(observationdatafile):
    """ Reads the observationdatafile as a time series and puts it into a list of floats.  This list is later compared
    to the floats parsed from the PySWMM reader of SWMM output.

    :param observationdatafile:
    :return time_difference:
    """
    with open(observationdatafile, 'r') as obs_file:
        # global contents
        contents = obs_file.readlines()
        # global obs_data, time_difference, obs_time
        obs_data = []
        obs_time = []
        for line in contents:
            linelist = list(line)
            if linelist[0] == ';' or linelist[0] == ' ' or len(list(line)) < 15:
                continue
            else:
                templine = line.split()
                if float(templine[-1]) < 0:
                    obs_data.append(0)
                else:
                    obs_data.append(float(templine[-1]))
                day_templine_preprocessing = line.replace(' ', ';')
                day_templine = re.split('[/|;|:|\t]', day_templine_preprocessing)
                month = int(day_templine[0])
                day = int(day_templine[1])
                year = int(day_templine[2])
                hour = int(day_templine[3])
                minute = int(day_templine[4])
                second = int(day_templine[5])
                if day_templine[6] == 'PM' and hour != 12:
                    hour = hour + 12
                elif day_templine[6] == 'AM' and hour == 12:
                    hour = 0
                obs_time.append(datetime.datetime(year, month, day, hour, minute, second))
        time_difference = obs_time[1] - obs_time[0]
    return time_difference, obs_data

def nashsutcliffe(hydrograph, obs_data):
    """Evaluates the NSE_m by computing the ratio of the difference of "obs_data" and "hydrograph" at each index to the
    difference between "obs_data" and its own average.
    Requires that readobservationfile() and PySWMM have both been called to initialize "obs_data" and "hydrograph".

    :return NSE_m: A float metric of the ratio between the predictive power of the model and the predictive power of
    simply the average of the observed data
    """
    average_obs = sum(obs_data)/len(obs_data)
    sum_sim_obs = 0
    sum_obs_obsave = 0
    for i in range(len(min(obs_data, hydrograph))-1):
        diff_sim_obs = (obs_data[i] - hydrograph[i])**2
        sum_sim_obs = sum_sim_obs + diff_sim_obs
        diff_obs_obsave = (obs_data[i] - average_obs)**2
        sum_obs_obsave = sum_obs_obsave + diff_obs_obsave
    mNSE = 1 - sum_sim_obs/sum_obs_obsave
    return mNSE

def objectivefunctions(trialfile, time_difference, obs_data, root=18):
    """ This function does the same thing as Par_objectivefunctions() with the exception being that it accepts a list
    of strings as an argument and not a string.  This means it is not parallelizable, but still used in the cross-over
    operation to determine which of the guesses is better.

    :param filelist:
    :param observationdatafile:
    :param distancefilename:
    :param root:
    :return:
    """
    
    hydrograph = []
    sim_time = []
    with Simulation(trialfile) as sim:
        node_object = Nodes(sim)
        root_location = node_object[root]
        simulation_timestep = time_difference.total_seconds()
        sim.step_advance(simulation_timestep)
        for step in sim:
            sim_time.append(sim.current_time)
            hydrograph.append(root_location.total_inflow)
    objFunc = nashsutcliffe(hydrograph, obs_data)
    return objFunc



if __name__ == '__main__':
    time_difference, obs_data = readobservationfile('./ref/Node14.dat')
    path = './ref/0210.inp'
    # path = '9/9_291.inp'
    res, nse2 = objectivefunctions(path, time_difference, obs_data, root='J14')
    print(res)
    print(nse2)