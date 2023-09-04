import os, sys
dir2 = os.path.abspath('')
dir1 = os.path.dirname(dir2)
if not dir1 in sys.path: 
    sys.path.append(dir1)
from math import sqrt
from collections import Counter
from datetime import datetime
import traceback

import torch
import tornado

from fuzzy_artmap.fuzzy_artmap_distributed_gpu import FuzzyArtmapGpuDistributed, FuzzyArtMapGpuWorker

import logging
logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

num_pats = 1_000
test_pats = 1_000
sq = 1;                         # Size of square
r = sq/sqrt(2*torch.pi);              # Radius of circle so it's half area of square
xcent = 0.5
ycent = 0.5                    # Centre of circle
xs = xcent*torch.ones((1,num_pats))
ys = ycent*torch.ones((1,num_pats))
train_rng = torch.random.manual_seed(12345) #torch.random.Generator(torch.random.PCG64(12345))
rng = torch.random.manual_seed(84562) #torch.random.Generator(torch.random.PCG64())
a = torch.concatenate((xs,ys)) + 0.5-torch.rand((2, num_pats))
bmat = ((a[0,:]-xcent)**2 + (a[1,:]-ycent)**2) > r**2
bmat = torch.vstack((bmat.long(), 1-bmat.long()))

xs = xcent*torch.ones((1,test_pats))
ys = ycent*torch.ones((1,test_pats))
test_set = torch.concatenate((xs,ys)) + 0.5-torch.rand((2, test_pats))
test_truth = ((test_set[0,:]-xcent)**2 + (test_set[1,:]-ycent)**2) > r**2
test_truth = torch.vstack((test_truth.long(), 1-test_truth.long()))


def get_traceback_string(e: Exception):
    if e is None:
        return "Passed exception is none!"
    if sys.version_info.minor >= 10:
        return ''.join(traceback.format_exception(e))
    else:
        return ''.join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__))


async def main():
    x = FuzzyArtmapGpuDistributed(4, 1, rho_a_bar = 0.0)
    await x.initialize_workers()
    start_time = datetime.now()
    print(start_time)
    for i in range(num_pats):
        test_input = torch.transpose(a[:, i, None], 0, 1)
        ground_truth = torch.transpose(bmat[:, i, None], 0, 1)
        complement_encoded_input = FuzzyArtMapGpuWorker.complement_encode(test_input)
        await x.train([complement_encoded_input], [ground_truth])

    out_test_point = torch.tensor(([0.115, 0.948],))
    encoded_test_point = FuzzyArtMapGpuWorker.complement_encode(out_test_point)
    prediction, fuzzy_match = await x.predict(encoded_test_point)
    print(prediction)

    in_test_point = torch.tensor(([0.262, 0.782],))
    encoded_test_point = FuzzyArtMapGpuWorker.complement_encode(in_test_point)
    prediction, fuzzy_match = await x.predict(encoded_test_point)
    print(prediction)

    test_predictions = Counter()
    for i in range(test_pats):
        test_input = torch.transpose(test_set[:, i, None], 0, 1)
        ground_truth = torch.transpose(test_truth[:, i, None], 0, 1)
        complement_encoded_input = FuzzyArtMapGpuWorker.complement_encode(test_input)        
        prediction, fuzzy_match = await x.predict(complement_encoded_input)
        correct = torch.all(prediction == ground_truth).item()
        test_predictions.update([correct])
    stop_time = datetime.now()
    print(f"elapsed: {stop_time-start_time}- {stop_time}")
    print(test_predictions)
    print(x.get_weight_a().shape)
    print(x.get_weight_ab().shape)
    print(torch.count_nonzero(x.get_weight_a()[:, 0] < 1, 0))

if __name__ == "__main__":
    try:
        tornado.ioloop.IOLoop.current().run_sync(main)
    except tornado.iostream.StreamClosedError as stream_error:
        trace_back_string = get_traceback_string(stream_error)
        LOGGER.fatal(f"At least one worker is terminated, cannot continue, exiting\ntraceback: {trace_back_string}")
        raise
    except Exception as e:
        trace_back_string = get_traceback_string(e)
        LOGGER.error(f"Error <{e}>\ntraceback: {trace_back_string}")
   