from model.generator import generate_network
from model.entities import Mode
from model.benchmark_instances import BenchmarkInstance
from model.benchmark_runner import ALGO_RUNNERS
from model.budget import EvalCounter
ALL_MODES = {Mode.WALK, Mode.BIKE, Mode.CAR, Mode.BUS, Mode.TRAM, Mode.METRO}
net = generate_network(n=20, nb=2, nt=1, nm=0, seed=201)
o, d = BenchmarkInstance('_','_',0,0,0,0,0,0,0).endpoints(net)
for name in ALGO_RUNNERS:
    counter = EvalCounter(budget=9000)
    front = ALGO_RUNNERS[name](net, o, d, ALL_MODES, 30, 9000, 1, counter)
    print(f'{name:16s} OK front={len(front)}')