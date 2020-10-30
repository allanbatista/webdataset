import os

import sys
import warnings
import time

import torch
import torch.multiprocessing as mp
from torch.utils.data import IterableDataset

import webdataset as wds
import queue
import threading

from . import filters


verbose = int(os.environ.get("MULTIDATASET_VERBOSE", 0))

timeout = float(os.environ.get("MULTIDATASET_TIMEOUT", 0.1))


def omp_warning():
    num_threads = int(os.environ.get("OMP_NUM_THREADS", "999999"))
    if num_threads >= 8:
        warnings.warn(f"set environment variale OMP_NUM_THREADS to something small")


def D(*args):
    if verbose:
        print(" ".join([str(x) for x in args]), file=sys.stderr)


def maybe_copy(a, pin_memory):
    if isinstance(a, torch.Tensor):
        if pin_memory:
            return a.pin_memory()
        else:
            return torch.new_tensor(a)
    else:
        return a


def copy_and_delete_tensors(sample, pin_memory=True):
    if isinstance(sample, (list, tuple)):
        result = tuple(maybe_copy(a, pin_memory) for a in sample)
        for a in sample:
            del a
    else:
        result = {k: maybe_copy(a, pin_memory) for k, a in sample.items()}
        for _, a in sample.items():
            del a
    return result


def wait(output_queue, prefetch):
    if prefetch <= 0:
        return None

    while output_queue.qsize() >= prefetch:
        time.sleep(0.1)


def _parallel_job(dataset, i, n, prefetch, output_queue):
    D("job", i, "started")
    dataset.shard_selection = lambda x: x[i::n]
    for cpu_sample in dataset:
        output_queue.put(cpu_sample)
        wait(output_queue, prefetch)

    D("job", i, "waiting")
    while output_queue.qsize() > 0:
        time.sleep(1.0)
    output_queue.close()
    D("job", i, "done", getattr(dataset, "sample_urls", None))


class MultiDatasetIterator(IterableDataset):
    def __init__(
        self, dataset=None, workers=4, output_size=100, pin_memory=True, prefetch=-1
    ):
        IterableDataset.__init__(self)
        omp_warning()
        self.output_queue = mp.Queue(output_size)
        self.pin_memory = pin_memory
        self.jobs = []
        for i in range(workers):
            job = mp.Process(
                target=_parallel_job,
                args=(dataset, i, workers, prefetch, self.output_queue),
                daemon=True,
            )
            self.jobs.append(job)
            job.start()
        D("started")

    def __iter__(self):
        return self

    def __next__(self):
        while len(self.jobs) > 0:
            try:
                result = self.output_queue.get(True, timeout=timeout)
                assert isinstance(result, (tuple, list, dict))
                if self.pin_memory:
                    result = copy_and_delete_tensors(result)
                return result
            except queue.Empty:
                D("queue empty")
                if all(job.exitcode is not None for job in self.jobs):
                    break
        D("all done")
        for job in self.jobs:
            job.join()
        D("all joined")
        raise StopIteration()

    def terminate(self):
        for job in self.jobs:
            if job.is_alive():
                job.terminate()

        time.sleep(1.0)
        for job in self.jobs:
            job.join()

    def __del__(self):
        self.terminate()


class MultiDataset(IterableDataset, wds.Pipeline):
    """MultiDataset is an experimental, generalized, pipeline-based alternative to DataLoader.

    The DataLoader class is complex and has some problems, in particular when
    it comes to working with IterableDatasets. MultiDataset is an experimental
    class showing what DataLoader might be replaced with in the future.

    Among other differences, MultiDataset handles splitting of samples among
    workers differently from DataLoader, and it also handles determining
    dataset length differently.

    So, for now, you want to use DataLoader if your training framework
    requires it, but you will have to deal with the limitations in DataLoader
    for IterableDataset. On the other hand, MultiDataset is a good choice in
    containers (since it doesn't use shared memory) or if you want a simpler
    way of controlling the assignment of shards to processes.
    """

    def __init__(
        self, dataset, workers=4, output_size=10000, nominal=None, pin_memory=True, prefetch=-1
    ):
        wds.Pipeline.__init__(self)
        D("dataset", dataset)
        self.kw = dict(
            dataset=dataset,
            workers=workers,
            output_size=output_size,
            pin_memory=pin_memory,
            prefetch=prefetch
        )
        self.nominal = nominal

    def __iter__(self):
        D("iter called")
        src = MultiDatasetIterator(**self.kw)
        return filters.pipeline(src, *self.pipeline)

    def __len__(self):
        return self.nominal


def to_device(sample, device):
    if isinstance(sample, (list, tuple)):
        return tuple(to_device(value, device) for value in sample)
    elif isinstance(sample, (dict,)):
        return {k: to_device(value, device) for k, value in sample.items()}
    else:
        return sample.to(device, non_blocking=True)


def reader(loader):
    try:
        for sample in loader.dataloader:
            if loader.finished:
                break
            data = to_device(sample, loader.device)
            if isinstance(sample, (list, tuple)):
                loader.queue.put(data)
            else:
                loader.queue.put((data,))
            del sample
            # waiting
            while (loader.queue.qsize() >= loader.prefetch_gpu) and not loader.finished:
                time.sleep(0.1)
    except Exception as e:
        loader.exception = e
    finally:
        loader.finished = True


class FasterMultiIterator:
    def __init__(self, dataloader, prefetch_gpu=2, device='cuda'):
        self.dataloader = dataloader
        self.queue = queue.Queue()
        self.finished = False
        self.device = device
        self.prefetch_gpu = prefetch_gpu
        self.thread = threading.Thread(target=reader, args=(self,))
        self.thread.start()
        self.exception = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.exception is not None:
            print("raise")
            raise self.exception

        while True:
            try:
                return self.queue.get(True, timeout=0.1)
            except queue.Empty:
                if self.finished:
                    raise StopIteration()

    def __del__(self):
        self.finished = True

        while not self.queue.empty():
            self.queue.get_nowait()

        if self.thread.is_alive():
            self.thread.join()

        torch.cuda.empty_cache()


class FasterMultiDataset:
    """
    This class pre enqueue send batchs to GPU. Make it ready to be processed before step beginning.
    """

    def __init__(self, dataset, workers=4, output_size=10000,
                 nominal=None, pin_memory=True, prefetch=-1,
                 prefetch_gpu=2, device='cuda'):
        self.dataloader = MultiDataset(dataset, workers, output_size, nominal, pin_memory, prefetch)
        self.device = device
        self.prefetch_gpu = prefetch_gpu

    def __iter__(self):
        return FasterMultiIterator(self.dataloader, prefetch_gpu=self.prefetch_gpu, device=self.device)
