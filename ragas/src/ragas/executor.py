from __future__ import annotations

import asyncio
import logging
import threading
import typing as t
from dataclasses import dataclass, field

from tqdm.auto import tqdm

from ragas.run_config import RunConfig
from ragas.utils import batched

logger = logging.getLogger(__name__)


@dataclass
class ExecutionError:
    """
    Marker class for failed job execution.

    This is returned instead of np.nan to provide more context about the failure
    and to make error handling more explicit for downstream code.
    """

    exception: Exception
    job_index: int

    def __bool__(self) -> bool:
        """Return False to indicate this is an error result."""
        return False

    def __str__(self) -> str:
        return f"ExecutionError(job={self.job_index}, exception={self.exception})"


def is_event_loop_running() -> bool:
    """
    Check if an event loop is currently running.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False
    else:
        return loop.is_running()


def run(
    async_func: t.Union[
        t.Callable[[], t.Coroutine[t.Any, t.Any, t.Any]],
        t.Coroutine[t.Any, t.Any, t.Any],
    ],
) -> t.Any:
    """Run an async function in the current event loop or a new one if not running."""
    try:
        # Check if we're already in a running event loop
        loop = asyncio.get_running_loop()
        # If we get here, we're in a running loop - need nest_asyncio
        try:
            import nest_asyncio
        except ImportError as e:
            raise ImportError(
                "It seems like you're running this in a jupyter-like environment. "
                "Please install nest_asyncio with `pip install nest_asyncio` to make it work."
            ) from e

        nest_asyncio.apply()
        # Create the coroutine if it's a callable, otherwise use directly
        coro = async_func() if callable(async_func) else async_func
        return loop.run_until_complete(coro)

    except RuntimeError:
        # No running event loop, so we can use asyncio.run
        coro = async_func() if callable(async_func) else async_func
        return asyncio.run(coro)


def as_completed(
    coroutines: t.List[t.Coroutine], max_workers: int
) -> t.Iterator[asyncio.Future]:
    """
    Wrap coroutines with a semaphore if max_workers is specified.

    Returns an iterator of futures that completes as tasks finish.
    """
    if max_workers == -1:
        tasks = [asyncio.create_task(coro) for coro in coroutines]
    else:
        semaphore = asyncio.Semaphore(max_workers)

        async def sema_coro(coro):
            async with semaphore:
                return await coro

        tasks = [asyncio.create_task(sema_coro(coro)) for coro in coroutines]

    return asyncio.as_completed(tasks)


async def process_futures(
    futures: t.Iterator[asyncio.Future], pbar: t.Optional[tqdm] = None
) -> t.AsyncGenerator[t.Any, None]:
    """
    Process futures with optional progress tracking.

    Args:
        futures: Iterator of asyncio futures to process (e.g., from asyncio.as_completed)
        pbar: Optional progress bar to update

    Yields:
        Results from completed futures as they finish
    """
    # Process completed futures as they finish
    for future in futures:
        result = await future
        if pbar:
            pbar.update(1)
        yield result


class ProgressBarManager:
    """Manages progress bars for batch and non-batch execution."""

    def __init__(self, desc: str, show_progress: bool):
        self.desc = desc
        self.show_progress = show_progress

    def create_single_bar(self, total: int) -> tqdm:
        """Create a single progress bar for non-batch execution."""
        return tqdm(
            total=total,
            desc=self.desc,
            disable=not self.show_progress,
        )

    def create_nested_bars(self, total_jobs: int, batch_size: int):
        """Create nested progress bars for batch execution."""
        n_batches = (total_jobs + batch_size - 1) // batch_size

        overall_pbar = tqdm(
            total=total_jobs,
            desc=self.desc,
            disable=not self.show_progress,
            position=0,
            leave=True,
        )

        batch_pbar = tqdm(
            total=min(batch_size, total_jobs),
            desc=f"Batch 1/{n_batches}",
            disable=not self.show_progress,
            position=1,
            leave=False,
        )

        return overall_pbar, batch_pbar, n_batches

    def update_batch_bar(
        self, batch_pbar: tqdm, batch_num: int, n_batches: int, batch_size: int
    ):
        """Update batch progress bar for new batch."""
        batch_pbar.reset(total=batch_size)
        batch_pbar.set_description(f"Batch {batch_num}/{n_batches}")


@dataclass
class Executor:
    """
    Executor class for running asynchronous jobs with progress tracking and error handling.

    Attributes
    ----------
    desc : str
        Description for the progress bar
    show_progress : bool
        Whether to show the progress bar
    keep_progress_bar : bool
        Whether to keep the progress bar after completion
    jobs : List[Any]
        List of jobs to execute
    raise_exceptions : bool
        Whether to raise exceptions or log them
    batch_size : int
        Whether to batch (large) lists of tasks
    run_config : RunConfig
        Configuration for the run
    _nest_asyncio_applied : bool
        Whether nest_asyncio has been applied
    """

    desc: str = "Evaluating"
    show_progress: bool = True
    keep_progress_bar: bool = True
    jobs: t.List[t.Any] = field(default_factory=list, repr=False)
    raise_exceptions: bool = False
    batch_size: t.Optional[int] = None
    run_config: t.Optional[RunConfig] = field(default=None, repr=False)
    pbar: t.Optional[tqdm] = None
    _jobs_processed: int = field(default=0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def wrap_callable_with_index(
        self, callable: t.Callable, counter: int
    ) -> t.Callable:
        async def wrapped_callable_async(*args, **kwargs) -> t.Tuple[int, t.Any]:
            try:
                result = await callable(*args, **kwargs)
                return counter, result
            except Exception as e:
                if self.raise_exceptions:
                    raise e
                else:
                    logger.error(
                        "Exception raised in Job[%s]: %s - %s",
                        counter,
                        type(e).__name__,
                        str(e),
                        exc_info=True,
                    )
                return counter, ExecutionError(e, counter)

        return wrapped_callable_async

    def submit(
        self,
        callable: t.Callable,
        *args,
        name: t.Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Submit a job to be executed, wrapping the callable with error handling and indexing to keep track of the job index.
        """
        # Use _jobs_processed for consistent indexing across multiple runs
        with self._lock:
            callable_with_index = self.wrap_callable_with_index(
                callable, self._jobs_processed
            )
            self.jobs.append((callable_with_index, args, kwargs, name))
            self._jobs_processed += 1

    def clear_jobs(self) -> None:
        """Clear all submitted jobs and reset counter."""
        with self._lock:
            self.jobs.clear()
            self._jobs_processed = 0

    async def _process_jobs(self) -> t.List[t.Any]:
        """Execute jobs with optional progress tracking."""
        with self._lock:
            if not self.jobs:
                return []

            # Make a copy of jobs to process and clear the original list to prevent re-execution
            jobs_to_process = self.jobs.copy()
            self.jobs.clear()

        max_workers = (self.run_config or RunConfig()).max_workers
        results = []
        pbm = ProgressBarManager(self.desc, self.show_progress)

        if not self.batch_size:
            # Use external progress bar if provided, otherwise create one
            if self.pbar is None:
                with pbm.create_single_bar(len(jobs_to_process)) as internal_pbar:
                    await self._process_coroutines(
                        jobs_to_process, internal_pbar, results, max_workers
                    )
            else:
                await self._process_coroutines(
                    jobs_to_process, self.pbar, results, max_workers
                )
            return results

        # Process jobs in batches with nested progress bars
        await self._process_batched_jobs(jobs_to_process, pbm, max_workers, results)
        return results

    async def _process_batched_jobs(
        self, jobs_to_process, progress_manager, max_workers, results
    ):
        """Process jobs in batches with nested progress tracking."""
        batch_size = self.batch_size or len(jobs_to_process)
        batches = batched(jobs_to_process, batch_size)
        overall_pbar, batch_pbar, n_batches = progress_manager.create_nested_bars(
            len(jobs_to_process), batch_size
        )

        with overall_pbar, batch_pbar:
            for i, batch in enumerate(batches, 1):
                progress_manager.update_batch_bar(batch_pbar, i, n_batches, len(batch))

                # Create coroutines per batch
                coroutines = [
                    afunc(*args, **kwargs) for afunc, args, kwargs, _ in batch
                ]
                async for result in process_futures(
                    as_completed(coroutines, max_workers), batch_pbar
                ):
                    results.append(result)
                # Update overall progress bar for all futures in this batch
                overall_pbar.update(len(batch))

    async def _process_coroutines(self, jobs, pbar, results, max_workers):
        """Helper function to process coroutines and update the progress bar."""
        coroutines = [afunc(*args, **kwargs) for afunc, args, kwargs, _ in jobs]
        async for result in process_futures(
            as_completed(coroutines, max_workers), pbar
        ):
            results.append(result)

    async def aresults(self) -> t.List[t.Any]:
        """
        Execute all submitted jobs and return their results asynchronously.
        The results are returned in the order of job submission.

        This is the async entry point for executing async jobs when already in an async context.
        """
        results = await self._process_jobs()
        sorted_results = sorted(results, key=lambda x: x[0])
        return [r[1] for r in sorted_results]

    def results(self) -> t.List[t.Any]:
        """
        Execute all submitted jobs and return their results. The results are returned in the order of job submission.

        This is the main sync entry point for executing async jobs.
        """

        # Check if we're in Jupyter/IPython
        try:
            from IPython.core.getipython import get_ipython

            if get_ipython() is not None:
                raise RuntimeError(
                    "In Jupyter/IPython, use `await executor.aresults()` directly; this avoids the need for nest_asyncio."
                )
        except ImportError:
            pass

        async def _async_wrapper():
            return await self.aresults()

        return run(_async_wrapper)


def run_async_batch(
    desc: str,
    func: t.Callable,
    kwargs_list: t.List[t.Dict],
    batch_size: t.Optional[int] = None,
):
    """
    Provide functionality to run the same async function with different arguments in parallel.
    """
    run_config = RunConfig()
    executor = Executor(
        desc=desc,
        keep_progress_bar=False,
        raise_exceptions=True,
        run_config=run_config,
        batch_size=batch_size,
    )

    for kwargs in kwargs_list:
        executor.submit(func, **kwargs)

    return executor.results()
