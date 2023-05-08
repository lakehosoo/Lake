import pandas as pd
import numpy as np
import datetime
import joblib
from sklearn import svm
from RTDB import get_rtdb_data

def make_x_op(df):
    v_nap1 = df['FQI11001']*1000
    v_dsr1 = df['FQI11003']/df['FQI11001']
    v_cop1 = df['PI11056']
    v_cot1 = (df['TC11100']+df['TC11200']+df['TC11300']+df['TC11400']+df['TC11500']+df['TC11600']+df['TC11700']+df['TC11800'])/8
    v_nap2 = df['FQI12001']*1000
    v_dsr2 = df['FQI12003']/df['FQI12001']
    v_cop2 = df['PI12056']
    v_cot2 = (df['TC12100']+df['TC12200']+df['TC12300']+df['TC12400']+df['TC12500']+df['TC12600']+df['TC12700']+df['TC12800'])/8
    v_nc4 = (df['PC3'] + df['PC4'] + df['OC4']) / 100
    v_nc5 = df['PC5'] * 0.5393 / 100
    v_nc6 = df['PC6'] * 0.4712 / 100
    v_nc7 = df['PC7'] * 0.3638 / 100
    v_nc8 = df['PC8'] * 0.0077 / 100
    v_nc9 = (df['PC9'] + df['PC10'] + df['PC11'] + df['PC12+']) / 100
    v_cc5a = (df['NC5'] + df['OC5']) / 100	
    v_cc5b = (df['NC6'] + df['OC6']) * 0.6361 / 100	
    v_cc6 = (df['NC6'] + df['OC6']) * 0.3639 / 100	
    v_cc7 = (df['NC7'] + df['OC7']) / 100	
    v_cc8 = (df['NC8'] + df['OC8']) / 100	
    v_cc9 = (df['NC9'] + df['NC10'] + df['NC11']) / 100	
    v_ac6 = df['AC6'] / 100
    v_ac7 = df['AC7'] / 100
    v_ac8 = (df['AC8'] + df['OC9'] + df['OC10'] + df['OC11'] + df['AC9'] + df['AC10'] + df['AC11'] + df['AC12+']) / 100
    
    x_op1 = np.array([v_nap1, v_dsr1, v_cop1, v_cot1, v_nc4, v_nc5, v_nc6, v_nc7, v_nc8, v_nc9, v_cc5a, v_cc5b, v_cc6, v_cc7, v_cc8, v_cc9, v_ac6, v_ac7, v_ac8])
    x_op2 = np.array([v_nap2, v_dsr2, v_cop2, v_cot2, v_nc4, v_nc5, v_nc6, v_nc7, v_nc8, v_nc9, v_cc5a, v_cc5b, v_cc6, v_cc7, v_cc8, v_cc9, v_ac6, v_ac7, v_ac8])
    x_op1 = x_op1.reshape(1,-1)
    x_op2 = x_op2.reshape(1,-1)
    return x_op1, x_op2, df.index[0]

def run_srgt_dl(x_op, name_y):
    # Loading SPYRO Surrogate Model : ANN
    folder='model_dl_nap'; case='DL'; nlayer1=20; nlayer2=10
    
    file = folder+'/srgt_reg_'+case+'_'+name_y+f'_{nlayer1}x{nlayer2}_weights.dat'
    y_min, y_max = np.loadtxt(folder+'/srgt_reg_'+case+'_y_lim_'+name_y+'.dat')
    x_min = np.loadtxt(folder+'/srgt_reg_'+case+'_x_min.dat')
    x_max = np.loadtxt(folder+'/srgt_reg_'+case+'_x_max.dat')
    
    x_op_scaled = (x_op-x_min)/(x_max-x_min)

    nx = x_op.shape[1]
    nt = nx + 2*nlayer1 + 2*nlayer2 +1
    a1 = np.genfromtxt(file, skip_header=0,                      skip_footer=nt-nx)
    b1 = np.genfromtxt(file, skip_header=nx,                     skip_footer=nt-nx-nlayer1)
    a2 = np.genfromtxt(file, skip_header=nx+nlayer1,             skip_footer=nt-nx-2*nlayer1)
    b2 = np.genfromtxt(file, skip_header=nx+2*nlayer1,           skip_footer=nt-nx-2*nlayer1-nlayer2)
    a3 = np.genfromtxt(file, skip_header=nx+2*nlayer1+nlayer2,   skip_footer=nt-nx-2*nlayer1-2*nlayer2)
    b3 = np.genfromtxt(file, skip_header=nx+2*nlayer1+2*nlayer2, skip_footer=0)
    
    X1 = np.maximum(0, np.dot(x_op_scaled, a1) + b1)
    X2 = np.maximum(0, np.dot(X1, a2) + b2)
    calc_save = np.maximum(0, np.dot(X2, a3) + b3)*(y_max-y_min)+y_min
    
    return calc_save

def run_epm(x_op, name_fnc, name_pass, name_y):
    # Loading Error Prediction Model
    folder='model_err'
    
    x_min, x_max = np.loadtxt(folder+'/epm_'+name_fnc+'_'+name_pass+'_'+name_y+'_x_lim.dat', delimiter=",")
    x_op_scaled = (x_op-x_min)/(x_max-x_min)
    
    ml_meas_opt = joblib.load(folder+'/epm_meas_'+name_fnc+'_'+name_pass+'_'+name_y+'.joblib')
    ml_esti_opt = joblib.load(folder+'/epm_esti_'+name_fnc+'_'+name_pass+'_'+name_y+'.joblib')

    err_meas_min, err_meas_max = np.loadtxt(folder+'/epm_meas_'+name_fnc+'_'+name_pass+'_'+name_y+'.dat', delimiter=",")
    err_esti_min, err_esti_max = np.loadtxt(folder+'/epm_esti_'+name_fnc+'_'+name_pass+'_'+name_y+'.dat', delimiter=",")

    err_meas_pred_s = ml_meas_opt.predict(x_op_scaled)
    err_esti_pred_s = ml_esti_opt.predict(x_op_scaled)

    err_meas_pred = err_meas_pred_s*(err_meas_max-err_meas_min)+err_meas_min
    err_esti_pred = err_esti_pred_s*(err_esti_max-err_esti_min)+err_esti_min
    
    return err_meas_pred, err_esti_pred

def run_batch(x_op, name_fnc, name_pass, name_y):
    
    i = list_y.index(name_y)
    j = 2*i + int(name_pass)
    
    # Calculation by Surrogate + Error Prediction Model
    calc_save = run_srgt_dl(x_op, name_y)
    err_meas_pred, err_esti_pred = run_epm(x_op, name_fnc, name_pass, name_y)

    pred_meas = calc_save + err_meas_pred
    pred_esti = calc_save + err_esti_pred
    
    return pred_meas[0], pred_esti[0]

if __name__ == "__main__" :
    now = datetime.datetime.now()
    print('Started : '+now.strftime('%Y-%m-%d %H:%M:%S'))

    taglist=[]
    with open("ss_main_tag_list.txt", "r") as f:
      for line in f:
        element = line.strip()
        taglist.append(element)

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = get_rtdb_data(taglist,now,now,1,600)
    df.set_index('DateTime', inplace=True)
    df.columns = df.columns.str.replace("LAB_MFC_FUR_FEED_NAP_", "", regex=True)
    df.columns = df.columns.str.replace("712", "", regex=True)
    df.columns = df.columns.str.replace(".PV", "", regex=True)

    x_op1, x_op2, t_ = make_x_op(df)

    now = datetime.datetime.now()
    print('RTDB Fetched : '+now.strftime('%Y-%m-%d %H:%M:%S'))

    list_fnc = ['111','121']
    list_pass = ['1','2']
    list_y = ['H2','CH4','C2H4','C2H6','C3H6']

    for name_fnc in list_fnc:
        if name_fnc=='111': x_op=x_op1;
        elif name_fnc=='121': x_op=x_op2;
        for name_pass in list_pass:
            res1=[]; res2=[];
            for i in range(5):
                name_y = list_y[i]
                pred1, pred2 = run_batch(x_op, name_fnc, name_pass, name_y)
                res1.append(pred1); res2.append(pred2)

                now = datetime.datetime.now()
                print('Predicted : '+now.strftime('%Y-%m-%d %H:%M:%S'))
                #print('Prediction for '+name_fnc+'/'+name_pass+'/'+name_y+f' : {result}')
            
            result = [t_,name_fnc,name_pass]+res1+res2
            try:
                with open("ss_main_output.txt", 'r') as f:
                    oldlines = f.readlines()
            except:
                oldlines = []
            with open("ss_main_output.txt", 'w') as f:
                f.writelines(oldlines)
                f.writelines(','.join(str(s) for s in result)+'\n')
