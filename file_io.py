import linecache, numpy

def read_params(path):
    filecache = linecache.getlines(path) # [57:202], [206:351]
    tmp = [filecache[57+i].split() + filecache[206+i].split() for i in range(145)]
    params = [[
        tmp[i][4], 
        tmp[i][6], 
        tmp[i][9],
        tmp[i][10],
        tmp[i][11],
        tmp[i][12],
        tmp[i][13]] for i in range(145)]
    return numpy.array(params).astype('float32')

def edit_params(params, save_path, edit_path):
    filecache = linecache.getlines(edit_path)
    for i in range(145):
        p = [f'{x:.6g}' for x in params[i]]
        tmp1 = filecache[57+i].split()
        filecache[57+i] = f'{tmp1[0]:<16} {tmp1[1]:<16} {tmp1[2]:<13} {tmp1[3]:<11} {p[0]:<8s} {tmp1[5]:<11} {p[1]:<8s} {tmp1[7]:<8}   \n'
        tmp2 = filecache[206+i].split()
        filecache[206+i] = f'{tmp2[0]:<16} {p[2]:<10s} {p[3]:<10s} {p[4]:<10s} {p[5]:<10s} {p[6]:<10s} {tmp2[6]:<10}   \n'
    with open(save_path,'w') as f:
        f.writelines(filecache)

    
# mapping from [low_bound, high_bound] to [0,10]
def params_to_vec(params, boundry):
    assert len(boundry) == len(params[0])
    for param in params:
        for i in range(len(boundry)):
            lb = boundry[i][0] # low bound
            hb = boundry[i][1] # high bound
            param[i] = (param[i]-lb)/(hb-lb)*10
    return params.reshape(-1)


# mapping from [0,10] to [low_bound, high_bound] 
def vec_to_params(vec, boundry):
    params = vec.reshape(-1, len(boundry)).copy()
    for param in params:
        for i in range(len(boundry)):
            lb = boundry[i][0] # low bound
            hb = boundry[i][1] # high bound
            param[i] = param[i]*(hb-lb)/10+lb
    return params



if __name__ == '__main__':
    BOUNDRY = [
        [0,     100],
        [0,     0.3],
        [0.011, 0.015],
        [0.014, 0.8],
        [1.27,  2.56],
        [2.56,  7.62],
        [40,    85]
    ]
    params = read_params('./0210.inp')
    # print(type(params[0][0]))
    b = numpy.array(BOUNDRY).astype('float32')
    vec = params_to_vec(params, b)
    print(vec[0])
    params = vec_to_params(vec, b)
    print(params[0])
    edit_params(params, './a.inp', './0210.inp')
