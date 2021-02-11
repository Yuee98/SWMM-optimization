import cma, numpy, os, pathlib, pickle, logging
from cma.constraints_handler import BoundTransform
from file_io import read_params, edit_params, vec_to_params, params_to_vec
from obj import objectivefunctions, readobservationfile
from matplotlib import pyplot as plt
import multiprocessing
global kwargs 
kwargs = {}
BOUNDRY = [
        [0,     100],
        [0,     3.25], # 初始值最大3.25，上限我改成了3.25
        [0.011, 0.015],
        [0.014, 0.8],
        [1.27,  2.56],
        [2.56,  7.62],
        [40,    85]
    ]
POPULATION_SIZE = 145*7 # 每轮生成多少个候选，最少145*7=维度
EPOCHS = 100 # 轮数
NUM_WORKERS = 4 # 并行计算数量，一般等于CPU核心，机械硬盘4大概到读写上限了，固态硬盘可以更高

def plot(max, average, save_path):
    plt.clf()
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.plot(max, 'r')
    plt.plot(average,'c')
    plt.savefig(save_path)

def work(i, vec):
    global kwargs 
    boundry = numpy.array(BOUNDRY).astype('float32')
    params = vec_to_params(vec, boundry)
    save_path = kwargs['path']/str(kwargs['ep'])/f'{kwargs["ep"]}_{i}.inp'
    edit_params(params, save_path, './ref/0210.inp')
    result = objectivefunctions(str(save_path), kwargs['time_difference'], kwargs['obs_data'], root='J14')
    logging.info(f'ep: {kwargs["ep"]} | solution: {kwargs["ep"]}_{i} | mNSE: {result}')
    return result


def main():
    logging.basicConfig(level=logging.INFO, filename='./optimize.log',
                    format='%(asctime)s: %(message)s')


    boundry = numpy.array(BOUNDRY).astype('float32')
    init_params = read_params('./ref/0210.inp')
    init_vec = params_to_vec(init_params, boundry)

    es = cma.CMAEvolutionStrategy(init_vec, 0.5, inopts={
        'BoundaryHandler': BoundTransform,
        'bounds': [[0], [10]]})

    time_difference, obs_data = readobservationfile('./ref/Node14.dat')
    path = pathlib.Path('./')
    if not (path/'fig').exists():
        os.mkdir(path/'fig')
    if not (path/'save').exists():
        os.mkdir(path/'save')
    Max, Avg = [], []
    for ep in range(EPOCHS):
        logging.info('==='*20+f'ep:{ep}'+'==='*20)
        if not (path/str(ep)).exists():
            os.mkdir(path/str(ep))

        candidates = es.ask(POPULATION_SIZE)

        global kwargs 
        kwargs = {
            'ep': ep,
            'path': path,
            'time_difference': time_difference, 
            'obs_data': obs_data
        }
        with multiprocessing.Pool(NUM_WORKERS) as pool:
            results = pool.starmap(work, enumerate(candidates))
        results = list(results)
        results = numpy.array(results)
        Max.append(results.max())
        Avg.append(results.mean())
        index = results.argmax()
        logging.info(f'ep {ep} best {ep}_{index}: {results[index]}')
        plot(Max, Avg, path/'fig'/f'{ep}.png')
        with open(path/'save'/f'{ep}.pkl', 'wb') as f:
            pickle.dump((es, candidates, results), f)

        rewards = - results
        es.tell(candidates, rewards)

        for f in (path/str(ep)).iterdir():
            if str(f) != f'{ep}/{ep}_{index}.inp':
                os.remove(f)

if __name__ == '__main__':
    main()



    
