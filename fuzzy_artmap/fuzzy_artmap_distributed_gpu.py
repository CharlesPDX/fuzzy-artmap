# Based on 
# Carpenter, G. A., Grossberg, S., Markuzon, N., Reynolds, J. H. and Rosen, D. B. (1992)
# "Fuzzy ARTMAP: A Neural Network Architecture for Incremental Supervised Learning of Analog Multidimensional Maps"
# IEEE Transactions on Neural Networks, Vol. 3, No. 5, pp. 698-713.

import cProfile
import pstats

import gc
import sys
import asyncio
import socket
import pickle
import argparse
import math
import struct
from enum import Enum, auto
import traceback
from typing import List
from datetime import datetime
from pathlib import Path
import logging
logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

file_logging_handler = logging.FileHandler('fuzzy_artmap_gpu_distributed.log')
file_logging_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_logging_handler.setFormatter(file_logging_format)
logger.addHandler(file_logging_handler)


import torch

from tornado.tcpserver import TCPServer
from tornado.tcpclient import TCPClient
from tornado.iostream import StreamClosedError
import tornado.ioloop

from .distributed_fuzzy_artmap_commands import DistributedFuzzyArtmapCommands


class ProcessingMode(Enum):
    local = auto()
    distributed = auto()

class FuzzyArtmapGpuDistributed:
    def __init__(self, 
                 f1_size: int = 10, 
                 f2_size: int = 10, 
                 number_of_categories: int = 2, 
                 rho_a_bar = 0, 
                 scheduler_address="127.0.0.1:5000", 
                 max_nodes = None, 
                 use_cuda_if_available = False, 
                 committed_beta = 0.75, 
                 mode: ProcessingMode = ProcessingMode.local): #TODO: add beta_ab for max nodes mode
        
        if rho_a_bar < 0.0 or rho_a_bar > 1.0:
            raise ValueError(f"rho_a_bar must be between 0.0 and 1.0, received {rho_a_bar}")
        
        if committed_beta < 0.0 or committed_beta > 1.0:
            raise ValueError(f"committed_beta must be between 0.0 and 1.0, received {committed_beta}")

        self.rho_a_bar = rho_a_bar  # Baseline vigilance for ARTa, in range [0,1]
        self.committed_beta = committed_beta
        self.f2_size = f2_size
        self.f1_size = f1_size
        self.number_of_categories = number_of_categories
        self.max_nodes = max_nodes
        self.use_cuda_if_available = use_cuda_if_available
        self.client = FuzzyArtmapWorkerClient(scheduler_address)
        self.mode = mode
        if self.mode == ProcessingMode.distributed or self.mode == ProcessingMode.local:
            self.training_fuzzy_artmap = FuzzyArtMapGpuWorker()
        else:
            raise ValueError(f"Invalid processing mode: {str(self.mode)}")
        
        self.weight_ab = self.training_fuzzy_artmap.weight_ab
    
    async def initialize_workers(self):
        logger.info(f"worker params: committed beta = {self.committed_beta}")
        if self.mode == ProcessingMode.distributed:
            logger.info("Getting workers")
            await self.client.get_workers()
            logger.info(f"Initializing {len(self.client.workers)} workers")
            params = [self.f1_size, self.f2_size, self.number_of_categories, self.rho_a_bar, self.max_nodes, self.use_cuda_if_available, self.committed_beta]
            await self.client.init_workers(params)
            self.training_fuzzy_artmap.init(self.f1_size, self.f2_size, self.number_of_categories, self.rho_a_bar, self.max_nodes, self.use_cuda_if_available, self.committed_beta)
        else:
            self.training_fuzzy_artmap.init(self.f1_size, self.f2_size, self.number_of_categories, self.rho_a_bar, self.max_nodes, self.use_cuda_if_available, self.committed_beta)
        logger.info("Workers initialized")

    
    async def train(self, input_vectors, class_vectors):
        if self.mode == ProcessingMode.distributed:
            io_loop = tornado.ioloop.IOLoop.current().asyncio_loop
            remote_fit = io_loop.create_task(self.client.fit([input_vectors, class_vectors]))
            local_fit = io_loop.run_in_executor(None, self.training_fuzzy_artmap.fit, input_vectors, class_vectors)
            await asyncio.gather(remote_fit, local_fit)
        elif self.mode == ProcessingMode.local:
            self.training_fuzzy_artmap.fit(input_vectors, class_vectors)

    async def predict(self, input_vector: torch.tensor):
        if self.mode == ProcessingMode.distributed:
            results = await self.client.predict(input_vector)
        elif self.mode == ProcessingMode.local:
            results = self.training_fuzzy_artmap.predict(input_vector)
        return results

    def save_model(self, descriptor):
        return self.training_fuzzy_artmap.save_model(descriptor)
    
    def load_model(self, model_path):
        if self.mode == ProcessingMode.distributed:
            raise NotImplementedError("Cannot currently load a model when running distributed mode.")
        loaded_f1_size, loaded_f2_size, parameters = self.training_fuzzy_artmap.load_model(model_path)
        self.f1_size = loaded_f1_size
        self.f2_size = loaded_f2_size
        self.number_of_categories = parameters["number_of_categories"]
        self.rho_a_bar =  parameters["rho_a_bar"]
        self.max_nodes = parameters["max_nodes"]
        self.use_cuda_if_available = parameters["use_cuda_if_available"]
        self.committed_beta = parameters["committed_beta"]
    
    def get_number_of_nodes(self):
        return self.training_fuzzy_artmap.weight_ab.shape[0]

    def get_number_of_increases(self):
        return self.training_fuzzy_artmap.number_of_increases

    def get_increase_size(self):
        return self.training_fuzzy_artmap.node_increase_step
    
    def get_committed_nodes(self):
        return ",".join([str(n) for n in self.training_fuzzy_artmap.committed_nodes])
    
    def get_weight_a(self):
        return self.training_fuzzy_artmap.weight_a
    
    def get_weight_ab(self):
        return self.training_fuzzy_artmap.weight_ab

class FuzzyArtMapGpuWorker:
    def __init__(self):
        self.alpha = 0.001  # "Choice" parameter > 0. Set small for the conservative limit (Fuzzy AM paper, Sect.3)
        self.weight_a = None
        self.weight_ab = None
        self.device = None
        self.input_vector_sum = None
        self.committed_nodes = set()
        self.updated_nodes = set()
        self.node_increase_step = 50 # number of F2 nodes to add when required
        self.number_of_increases = 0

        self.beta = 1  # Learning rate. Set to 1 for fast learning
        self.committed_beta = 0.75
        self.beta_ab = 1 #ab learning rate
        
        
        self.rho_ab = 0.95          # Map field vigilance, in [0,1]
        self.epsilon = 0.001        # Fab mismatch raises ARTa vigilance to this much above what is needed to reset ARTa

        self.rho_a_bar = None  # Baseline vigilance for ARTa, in range [0,1]
        self.max_nodes = None

        self.A_and_w = None
        self.profiler = None

        self.dtype = torch.float
        self.parameters = {}

    def init(self, f1_size: int = 10, f2_size: int = 10, number_of_categories: int = 2, rho_a_bar = 0, max_nodes = None, use_cuda_if_available = False, committed_beta = 0.75):
        self.rho_a_bar = rho_a_bar  # Baseline vigilance for ARTa, in range [0,1]
        self.committed_beta = committed_beta
        self.max_nodes = max_nodes
        if use_cuda_if_available and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            if use_cuda_if_available:
                logger.warning("CUDA requested but not available, using CPU.")
        self.weight_a = torch.ones((f2_size, f1_size), device=self.device, dtype=self.dtype)
        self.input_vector_sum = f1_size / 2
        self.weight_ab = torch.ones((f2_size, number_of_categories), device=self.device, dtype=self.dtype)
        self.A_and_w = torch.empty(self.weight_a.shape, device=self.device, dtype=self.dtype)
        logger.info(f"f1_size: {f1_size}, f2_size:{f2_size}, committed beta = {self.committed_beta}")
        self.parameters["f1_size"] = f1_size
        self.parameters["f2_size"] = f2_size
        self.parameters["number_of_categories"] = number_of_categories
        self.parameters["rho_a_bar"] = rho_a_bar
        self.parameters["max_nodes"] = max_nodes
        self.parameters["use_cuda_if_available"] = use_cuda_if_available
        self.parameters["committed_beta"] = committed_beta
        # self.profiler = cProfile.Profile()

    def _resonance_search_vector(self, input_vector: torch.tensor, already_reset_nodes: List[int], rho_a: float):
        # self.profiler.enable()
        resonant_a = False
        N, S, T = self.calculate_activation(input_vector)
        sorted_values, indices = torch.sort(T, stable=True, descending=True)
        all_membership_degrees = S / self.input_vector_sum
        T[already_reset_nodes] = torch.zeros((len(already_reset_nodes), ), dtype=self.dtype, device=self.device)
        while not resonant_a:
            for J in indices:
                if J.item() in already_reset_nodes:
                    continue

                if all_membership_degrees[J].item() >= rho_a or math.isclose(all_membership_degrees[J].item(), rho_a):
                    resonant_a = True
                    break
                else:
                    resonant_a = False
                    already_reset_nodes.append(indices[J].item())
                    T[indices[J].item()] = 0

            # Creating a new node if we've reset all of them
            if len(already_reset_nodes) >= N:                
                if self.max_nodes is None or self.max_nodes >= (N + self.node_increase_step):
                    self.weight_a = torch.vstack((self.weight_a, torch.ones((self.node_increase_step,  self.weight_a.shape[1]), device=self.device, dtype=self.dtype)))
                    self.weight_ab = torch.vstack((self.weight_ab, torch.ones((self.node_increase_step, self.weight_ab.shape[1]), device=self.device, dtype=self.dtype)))
                    self.A_and_w = torch.vstack((self.A_and_w, torch.empty((self.node_increase_step,  self.weight_a.shape[1]), device=self.device, dtype=self.dtype)))
                    self.number_of_increases += 1
                else:                   
                    self.rho_ab = 0
                    self.beta_ab = 0.75
                    self.rho_a_bar = 0
                    rho_a = self.rho_a_bar
                    logger.info(f"Maximum number of nodes reached, {len(already_reset_nodes)} - adjusting rho_ab to {self.rho_ab} and beta_ab to {self.beta_ab}")
                    already_reset_nodes.clear()
                N, S, T = self.calculate_activation(input_vector)
                sorted_values, indices = torch.sort(T, stable=True, descending=True)
                all_membership_degrees = S / self.input_vector_sum
                T[already_reset_nodes] = torch.zeros((len(already_reset_nodes), ), dtype=self.dtype, device=self.device)
                
        # self.profiler.disable()
        # stats = pstats.Stats(self.profiler).sort_stats('cumtime')
        # stats.print_stats()
        return J.item(), all_membership_degrees[J].item()

    def calculate_activation(self, input_vector):
        N = self.weight_a.shape[0]  # Count how many F2a nodes we have

        torch.minimum(input_vector.repeat(N,1), self.weight_a, out=self.A_and_w) # Fuzzy AND = min
        S = torch.sum(self.A_and_w, 1) # Row vector of signals to F2 nodes
        T = S / (self.alpha + torch.sum(self.weight_a, 1)) # Choice function vector for F2
        return N,S,T

    def train(self, input_vector: torch.tensor, class_vector: torch.tensor):
        rho_a = self.rho_a_bar # We start off with ARTa vigilance at baseline
        resonant_ab = False # Not resonating in the Fab match layer
        already_reset_nodes = [] # We haven't rest any ARTa nodes for this input pattern yet, maintain list between resonance searches of Fa
        
        class_vector = class_vector.to(self.device)
        input_vector = input_vector.to(self.device)
        class_vector_sum = torch.sum(class_vector, 1)
        while not resonant_ab:            
            J, x = self._resonance_search_vector(input_vector, already_reset_nodes, rho_a)
            
            z = torch.minimum(class_vector, self.weight_ab[J, None])
            
            resonance = torch.sum(z, 1)/class_vector_sum
            if resonance > self.rho_ab or math.isclose(resonance, self.rho_ab):
                resonant_ab = True
            else: 
                already_reset_nodes.append(J)
                rho_a = x + self.epsilon                
                if rho_a > 1.0:
                    rho_a = 1.0 - self.epsilon

        self.updated_nodes.add(J)
        if J in self.committed_nodes:
            beta = self.committed_beta
        else:
            beta = self.beta

        self.weight_a[J, None] = (beta * torch.minimum(input_vector, self.weight_a[J, None])) + ((1-beta) * self.weight_a[J, None])
        self.weight_ab[J, None] = (self.beta_ab * z) + ((1-self.beta_ab) * self.weight_ab[J, None])
        self.committed_nodes.add(J)

    def fit(self, input_vectors, class_vectors):
        # self.profiler = cProfile.Profile()
        # self.profiler.enable()
        number_of_increases_before_training = self.number_of_increases
        for vector_index, input_vector in enumerate(input_vectors):
            self.train(input_vector, class_vectors[vector_index])
        # self.profiler.disable()
        # stats = pstats.Stats(self.profiler).sort_stats('cumtime')
        # stats.print_stats()
        number_of_added_nodes = (self.number_of_increases - number_of_increases_before_training) * self.node_increase_step
        # logger.info(f"added {number_of_added_nodes} nodes, updated {len(self.updated_nodes)} nodes: {','.join([str(J) for J in self.updated_nodes])}")
        self.updated_nodes.clear()


    @staticmethod
    def complement_encode(original_vector: torch.tensor) -> torch.tensor:
        complement = 1-original_vector
        complement_encoded_value = torch.hstack((original_vector,complement))
        return complement_encoded_value

    def predict(self, input_vector: torch.tensor):
        rho_a = 0 # set ARTa vigilance to first match
        J, membership_degree = self._resonance_search_vector(input_vector, [], rho_a)
        
        # (Called x_ab in Fuzzy ARTMAP paper)
        return self.weight_ab[J, None], membership_degree # Fab activation vector & fuzzy membership value

    def save_model(self, descriptor):
        model_timestamp = datetime.now().isoformat().replace("-", "_").replace(":", "_").replace(".", "_")
        cleaned_descriptor = descriptor.replace("-", "_").replace(":", "_").replace(".", "_")
        model_path = f"famgd_{model_timestamp}_{cleaned_descriptor}.pt"
        torch.save((self.weight_a, self.weight_ab, self.committed_nodes, self.parameters), model_path)
        return model_path

    def load_model(self, model_path):
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"`{model_path}` was not found or is a directory")
        logger.info(f"Loading model from {model_path}")
        weight_a, weight_ab, committed_nodes, parameters = torch.load(model_path)

        loaded_f1_size = weight_a.shape[1]
        loaded_f2_size = weight_a.shape[0]
        logger.info(f"Parameter f1: {parameters['f1_size']}, f2: {parameters['f2_size']} - Actual f1: {loaded_f1_size}, f2: {loaded_f2_size}")
        self.init(loaded_f1_size, loaded_f2_size, parameters["number_of_categories"], parameters["rho_a_bar"], parameters["max_nodes"], parameters["use_cuda_if_available"], parameters["committed_beta"])
        self.weight_a = weight_a
        self.weight_ab = weight_ab
        self.committed_nodes = committed_nodes
        logger.info("Model loaded")
        return loaded_f1_size, loaded_f2_size, parameters


class FuzzyArtmapWorkerServer(TCPServer):
    def __init__(self, ssl_options = None, max_buffer_size = None, read_chunk_size = None) -> None:        
        super().__init__(ssl_options, max_buffer_size, read_chunk_size)
        self.model = None        
        self.commands = DistributedFuzzyArtmapCommands()
        # gc.disable()

    async def handle_stream(self, stream, address):
        while True:
            data_buffer = bytearray()
            try:
                data = await stream.read_until(self.commands.end_mark)
                expected_length = struct.unpack("I", data[1:5])[0]
                actual_length = len(data) - self.commands.protocol_overhead
                while actual_length != expected_length:
                    logger.debug(f"received {actual_length} so far, expected {expected_length} - waiting on remaining data")
                    data_buffer.extend(data)
                    data = await stream.read_until(self.commands.end_mark)
                    logger.debug(f"received {len(data)} extra")
                    actual_length += len(data)
                    if actual_length == expected_length:
                        logger.debug(f"expected data arrived")
                        data_buffer.extend(data)
                        data = data_buffer
                        break
                try:
                    await self.handle_data(data[:-3], stream)
                    data_buffer.clear()
                except Exception as e:
                    if sys.version_info.minor >= 10:
                        traceback_string = ''.join(traceback.format_exception(e))
                    else:
                        traceback_string = ''.join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__))
                    logger.error(f"error running {chr(data[0])} operation - {traceback_string}")
                    worker_id = struct.pack("I", self.worker_index)
                    error_bytes = traceback_string.encode("utf-8")
                    error_length = struct.pack("I", len(error_bytes))
                    await stream.write(self.commands.error_header + worker_id + error_length + error_bytes + self.commands.end_mark)
            except StreamClosedError:
                logger.error("connection closed")
                break
    
    async def handle_data(self, data, stream):
        logger.info(f"received header: {chr(data[0])}")
        expected_length = struct.unpack("I", data[1:5])[0]
        actual_length = len(data) - 5
        if actual_length != expected_length:
            logger.error(f"received {actual_length} - expected {expected_length}")
        
        if data[0] == self.commands.init_header[0]:
            self.model = FuzzyArtMapGpuWorker()
            gc.collect()
            self.worker_index = struct.unpack("I", data[5:9])[0]
            logger.info(f"worker_id: {self.worker_index}")
            init_params = pickle.loads(data[9:])
            
            self.model.init(*init_params)
            await stream.write(self.commands.end_mark)
            logger.info("init completed")

        elif data[0] == self.commands.predict_header[0]:
            # self.profiler = cProfile.Profile()
            # self.profiler.enable()
            doc_ids = pickle.loads(data[5:])
            results = self.model.predict_proba(doc_ids)
            pickled_results = pickle.dumps(results)
            results_length = struct.pack("I", len(pickled_results))
            await stream.write(self.commands.prediction_response_header + results_length + pickled_results + self.commands.end_mark)
            logger.info("predict completed")
            # self.profiler.disable()
            # stats = pstats.Stats(self.profiler).sort_stats('cumtime')
            # stats.print_stats()

        elif data[0] == self.commands.fit_header[0]:
            params = pickle.loads(data[5:])
            self.model.fit(*params)
            await stream.write(self.commands.end_mark)
            logger.info("training completed")
        
        else:
            print(data)

class FuzzyArtmapWorkerClient():
    def __init__(self, registrar_address) -> None:        
        self.host, self.port = registrar_address.split(":")
        self.commands = DistributedFuzzyArtmapCommands()
        self.workers = []
    
    async def get_workers(self):
        client = TCPClient()
        logger.info(f"connecting to registrar: {self.host}:{self.port}")
        stream = await client.connect(self.host, int(self.port))
        await stream.write(self.commands.get_worker_address_header)
        response = await stream.read_until(self.commands.end_mark)        
        for worker_address in response[:-3].decode("utf-8").split(","):
            worker_host, worker_port = worker_address.split(":")
            logger.info(f"connecting to worker: {worker_address}")
            worker_stream = await client.connect(worker_host, int(worker_port))
            self.workers.append(worker_stream)
    
    async def init_workers(self, params):
        logger.info("init workers entered")
        futures = []
        pickled_params = pickle.dumps(params)
        params_length = struct.pack("I", len(pickled_params) + 4)
        for worker_index, worker in enumerate(self.workers):
           futures.append(worker.write(self.commands.init_header + params_length + struct.pack("I", worker_index) + pickled_params + self.commands.end_mark))
        
        await asyncio.gather(*futures)
        await self.get_responses()
        logger.info("init workers completed")

    async def fit(self, params):
        logger.info("starting remote fit")
        futures = []
        pickled_params = pickle.dumps(params)
        params_size = struct.pack("I", len(pickled_params))
        for worker in self.workers:
           futures.append(worker.write(self.commands.fit_header + params_size + pickled_params + self.commands.end_mark))

        await asyncio.gather(*futures)
        await self.get_responses()
        logger.info("exiting remote fit")
    
    async def predict(self, input_vector):
        logger.info("predict entered")
        pickled_input_vector = pickle.dumps(input_vector)
        params_size = struct.pack("I", len(pickled_input_vector))
        futures = []
        for worker in self.workers:
            futures.append(worker.write(self.commands.predict_header + params_size + pickled_input_vector + self.commands.end_mark))
        
        await asyncio.gather(*futures)
        worker_results = await self.get_responses()
        results = []
        for result in worker_results:
            results.extend(pickle.loads(result[5:-3]))
        logger.info("predict completed")
        return results

    def check_response(self, response):
        if len(response) != 3:
            if response[0] == self.commands.prediction_response_header[0] and response[-3:] == self.commands.end_mark:
                return

            if response[0] == self.commands.error_header[0]:
                worker_id = struct.unpack("I", response[1:5])[0]
                error_stop_index = struct.unpack("I", response[5:9])[0] + 9
                error_message = response[9:error_stop_index].decode("utf-8")
                exception_message = f"worker {worker_id} returned error {error_message}"
                logger.error(exception_message)
                raise Exception(exception_message)
            else:
                raise Exception(f"unknown worker error")

    async def get_responses(self):
        response_futures = []
        for worker in self.workers:
            response_futures.append(worker.read_until(self.commands.end_mark))
        
        responses = await asyncio.gather(*response_futures)
        results = []
        for response in responses:
            self.check_response(response)
            results.append(response)
        return results

async def register_worker():
    if args.localhost:
        data = f"r{socket.gethostbyname('localhost')}:{args.port}"
    else:        
        hostname = socket.gethostname()
        local_ip_address = socket.gethostbyname(hostname)
        if local_ip_address == "127.0.0.1":
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 1))  # connect() for UDP doesn't send packets
            local_ip_address = s.getsockname()[0]
        data = f"r{local_ip_address}:{args.port}"
    
    client = TCPClient()
    host, port = args.registrar.split(":")
    logger.info(f"connecting to registrar: {args.registrar}")
    stream = await client.connect(host, int(port))
    logger.info(f"registering worker at {data}")
    data = data.encode("utf-8")
    await stream.write(data)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-r", "--registrar", help="ip:port of the registrar server", required=True)
    arg_parser.add_argument("-l", "--localhost", help="report localhost as the worker address", action=argparse.BooleanOptionalAction)
    arg_parser.add_argument("-p", "--port", help="worker listener port, override default 48576", default="48576")

    args = arg_parser.parse_args()
    
    tornado.ioloop.IOLoop.current().run_sync(register_worker)
    server = FuzzyArtmapWorkerServer()
    logger.info('Starting the server...')
    server.listen(int(args.port))
    tornado.ioloop.IOLoop.current().start()
    logger.info('Server has shut down.')