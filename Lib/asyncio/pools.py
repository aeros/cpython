"""Support for high-level asynchronous pools in asyncio."""

__all__ = 'ThreadPool',


import concurrent.futures
import functools
import threading
import os

from . import events
from . import exceptions
from . import futures


class AbstractPool:
    """Abstract base class for asynchronous pools."""

    async def start(self):
        raise NotImplementedError

    async def __aenter__(self):
        await self.start()
        return self

    async def aclose(self):
        raise NotImplementedError

    async def __aexit__(self, exc_type, exc_value, exc_traceback):
        await self.aclose()

    async def run(self, func, *args, **kwargs):
        """Asynchronously run function *func* using the pool.

        Return a future, representing the eventual result of *func*.
        """
        raise NotImplementedError


class ThreadPool(AbstractPool):
    """Asynchronous threadpool for running IO-bound functions.

    Directly calling an IO-bound function within the main thread will block
    other operations from occurring until it is completed. By using a
    threadpool, several IO-bound functions can be ran concurrently within
    their own threads, without blocking other operations.

    The optional argument *concurrency* sets the number of threads within the
    threadpool. If *concurrency* is `None`, the maximum number of threads will
    be used; based on the number of CPU cores.

    This threadpool is intended to be used as an asynchronous context manager,
    using the `async with` syntax, which provides automatic initialization and
    finalization of resources. For example:

    def blocking_io():
        print("start blocking_io")
        with open('/dev/urandom', 'rb') as f:
            f.read(100_000)
        print("blocking_io complete")

    def other_io():
        print("start other_io")
        with open('/dev/zero', 'rb') as f:
            f.read(10)
        print("other_io complete")

    async def main():
        with asyncio.ThreadPool() as pool:
            await asyncio.gather(pool.run(blocking_io),
                                 pool.run(other_io))


    asyncio.run(main())
    """

    def __init__(self, concurrency=None):
        if concurrency is None:
            concurrency = min(32, (os.cpu_count() or 1) + 4)

        self.concurrency = concurrency
        self.closed = False
        self.running = False
        self._shutting_down = False
        self._loop = None
        self._pool = None

    async def start(self):
        self._loop = events.get_running_loop()
        await self.spawn_threadpool()

    async def __aenter__(self):
        await self.start()
        return self

    async def aclose(self):
        await self.shutdown_threadpool()

    async def __aexit__(self, exc_type, exc_value, exc_traceback):
        await self.aclose()

    async def run(self, func, *args, **kwargs):
        if self.closed:
            raise RuntimeError(f"unable to run {func!r}, "
                               "threadpool is closed")

        if not self.running:
            raise RuntimeError(f"unable to run {func!r}, "
                               "threadpool is not running")

        func_call = functools.partial(func, *args, **kwargs)
        executor = self._pool
        return await futures.wrap_future(
            executor.submit(func_call), loop=self._loop)

    async def spawn_threadpool(self, concurrency=None):
        """Schedule the spawning of the threadpool.

        Asynchronously spawns a threadpool with *concurrency* threads.

        Note that there is no need to call this method explicitly when using
        `asyncio.ThreadPool` as a context manager.
        """
        if self.running:
            raise RuntimeError("threadpool is already running")

        if self.closed:
            raise RuntimeError("threadpool is closed")

        future = self._loop.create_future()
        thread = threading.Thread(target=self._do_spawn,
                                  args=(future, concurrency))

        thread.start()
        try:
            await future
        finally:
            thread.join()

    def _do_spawn(self, future, concurrency):
        try:
            self._pool = concurrent.futures.ThreadPoolExecutor(
                                                max_workers=concurrency)
            self.running = True
            self._loop.call_soon_threadsafe(future.set_result, None)
        except Exception as ex:
            self._loop.call_soon_threadsafe(future.exception, ex)

    async def shutdown_threadpool(self):
        """Schedule the shutdown of the threadpool.

        Asynchronously joins all of the threads in the threadpool.

        Note that there is no need to call this method explcitly when using
        `asyncio.ThreadPool` as a context manager.
        """
        if self.closed:
            raise RuntimeError("threadpool is already closed")

        future = self._loop.create_future()
        thread = threading.Thread(target=self._do_shutdown, args=(future,))
        thread.start()
        try:
            await future
        finally:
            thread.join()

    def _do_shutdown(self, future):
        try:
            executor = self._pool
            executor.shutdown(wait=True)
            self.closed = True
            self.running = False
            self._loop.call_soon_threadsafe(future.set_result, None)
        except Exception as ex:
            self._loop.call_soon_threadsafe(future.exception, ex)
