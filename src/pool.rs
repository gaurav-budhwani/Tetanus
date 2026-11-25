//! Persistent cache-line-aligned thread pool for low-latency matrix-vector products.
//!
//! Why not `std::thread::scope` per matmul?
//! A 24-layer Transformer executes ~170 matmuls per generated token. Spawning and joining
//! threads inside each matmul costs ~20-50 us of OS overhead, burning milliseconds per token.
//!
//! Instead, this pool keeps worker threads alive across the session with a two-phase wait:
//!   1. Short atomic spin-wait (~2048 iterations) so back-to-back layer matmuls wake in <1 us.
//!   2. `Condvar` sleep between prompts so idle CPU stays at 0%.
//!
//! Atomic counters are padded to 128 bytes on Apple Silicon (M1-M4 cache line size) and
//! 64 bytes on x86_64 to prevent false sharing across cores.

use std::ffi::c_void;
use std::sync::atomic::{fence, AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

#[cfg(target_arch = "aarch64")]
#[repr(align(128))]
struct CachePadded<T>(T);

#[cfg(not(target_arch = "aarch64"))]
#[repr(align(64))]
struct CachePadded<T>(T);

struct JobDesc {
    func: unsafe fn(*const c_void, usize, usize),
    ctx: *const c_void,
    total_threads: usize,
}

// Safety: `JobDesc` is only dereferenced by workers while the dispatcher thread blocks on
// `done_count`, ensuring the pointed-to closure outlives all worker accesses.
unsafe impl Sync for JobDesc {}

struct SharedState {
    epoch: CachePadded<AtomicU64>,
    done_count: CachePadded<AtomicUsize>,
    parked_count: CachePadded<AtomicUsize>,
    active_job: AtomicPtr<JobDesc>,
    shutdown: AtomicBool,
    wait_lock: Mutex<()>,
    wait_cv: Condvar,
}

/// Persistent worker pool where worker `0` is always the calling thread.
pub struct ThreadPool {
    num_threads: usize,
    shared: Arc<SharedState>,
    handles: Vec<JoinHandle<()>>,
}

impl ThreadPool {
    /// Creates a pool with `num_threads` total execution slots (`num_threads - 1` background threads).
    pub fn new(num_threads: usize) -> Self {
        let num_threads = num_threads.max(1);
        let shared = Arc::new(SharedState {
            epoch: CachePadded(AtomicU64::new(0)),
            done_count: CachePadded(AtomicUsize::new(0)),
            parked_count: CachePadded(AtomicUsize::new(0)),
            active_job: AtomicPtr::new(std::ptr::null_mut()),
            shutdown: AtomicBool::new(false),
            wait_lock: Mutex::new(()),
            wait_cv: Condvar::new(),
        });

        let mut handles = Vec::with_capacity(num_threads - 1);
        for worker_id in 1..num_threads {
            let s = Arc::clone(&shared);
            let handle = thread::Builder::new()
                .name(format!("tetanus-w{worker_id}"))
                .spawn(move || worker_loop(worker_id, &s))
                .expect("failed to spawn worker thread");
            handles.push(handle);
        }

        Self {
            num_threads,
            shared,
            handles,
        }
    }

    #[inline]
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// Runs `f(worker_id, num_threads)` across all pool threads and blocks until all finish.
    /// When `num_threads == 1` or work is tiny, runs inline without touching atomics.
    pub fn parallel_for<F>(&self, f: F)
    where
        F: Fn(usize, usize) + Sync,
    {
        if self.num_threads == 1 {
            f(0, 1);
            return;
        }

        unsafe fn trampoline<F: Fn(usize, usize) + Sync>(
            ctx: *const c_void,
            worker_id: usize,
            total_threads: usize,
        ) {
            // Safety: `ctx` points to stack-pinned `f` on the caller thread until all workers signal done.
            let closure = unsafe { &*(ctx.cast::<F>()) };
            closure(worker_id, total_threads);
        }

        let mut job = JobDesc {
            func: trampoline::<F>,
            ctx: (&f as *const F).cast::<c_void>(),
            total_threads: self.num_threads,
        };

        // Publish job pointer before bumping `epoch` (`SeqCst` pairs with worker Dekker check).
        self.shared.done_count.0.store(0, Ordering::Relaxed);
        self.shared
            .active_job
            .store(&mut job as *mut JobDesc, Ordering::Relaxed);
        self.shared.epoch.0.fetch_add(1, Ordering::SeqCst);

        // Only take the mutex to wake parked threads when at least one worker actually slept.
        if self.shared.parked_count.0.load(Ordering::SeqCst) > 0 {
            let _guard = self.shared.wait_lock.lock().unwrap();
            self.shared.wait_cv.notify_all();
        }

        // Execute shard 0 on the calling thread while background workers run shards 1..N.
        f(0, self.num_threads);

        let expected = self.num_threads - 1;
        while self.shared.done_count.0.load(Ordering::Acquire) < expected {
            std::hint::spin_loop();
        }

        // Clear pointer strictly after all workers have finished their shard.
        self.shared
            .active_job
            .store(std::ptr::null_mut(), Ordering::Relaxed);
        fence(Ordering::Acquire);
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        if self.handles.is_empty() {
            return;
        }
        {
            let _guard = self.shared.wait_lock.lock().unwrap();
            self.shared.shutdown.store(true, Ordering::SeqCst);
            self.shared.wait_cv.notify_all();
        }
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

fn worker_loop(worker_id: usize, shared: &SharedState) {
    let mut local_epoch: u64 = 0;

    loop {
        // Phase 1: brief spin to absorb inter-layer gap without a kernel syscall.
        let mut new_epoch = shared.epoch.0.load(Ordering::Acquire);
        if new_epoch == local_epoch {
            for _ in 0..2048 {
                if shared.shutdown.load(Ordering::Relaxed) {
                    return;
                }
                std::hint::spin_loop();
                new_epoch = shared.epoch.0.load(Ordering::Acquire);
                if new_epoch != local_epoch {
                    break;
                }
            }
        }

        // Phase 2: park on Condvar if still idle (Dekker-ordered against `epoch`).
        if new_epoch == local_epoch {
            let mut guard = shared.wait_lock.lock().unwrap();
            shared.parked_count.0.fetch_add(1, Ordering::SeqCst);
            loop {
                if shared.shutdown.load(Ordering::SeqCst) {
                    shared.parked_count.0.fetch_sub(1, Ordering::SeqCst);
                    return;
                }
                let latest = shared.epoch.0.load(Ordering::SeqCst);
                if latest != local_epoch {
                    new_epoch = latest;
                    break;
                }
                guard = shared.wait_cv.wait(guard).unwrap();
            }
            shared.parked_count.0.fetch_sub(1, Ordering::SeqCst);
        }

        local_epoch = new_epoch;
        let job_ptr = shared.active_job.load(Ordering::Acquire);
        if !job_ptr.is_null() {
            // Safety: `job_ptr` remains valid until we increment `done_count` below.
            unsafe {
                let job = &*job_ptr;
                (job.func)(job.ctx, worker_id, job.total_threads);
            }
            shared.done_count.0.fetch_add(1, Ordering::Release);
        }
    }
}

/// Splits `total` work items evenly across `num_threads` contiguous shards.
#[inline]
pub fn shard_range(total: usize, worker_id: usize, num_threads: usize) -> std::ops::Range<usize> {
    let base = total / num_threads;
    let rem = total % num_threads;
    let start = worker_id * base + worker_id.min(rem);
    let len = base + usize::from(worker_id < rem);
    start..(start + len)
}

/// Detects the ideal default worker thread count for memory-bandwidth-bound CPU inference.
///
/// On Apple Silicon (M1-M4), `hw.perflevel0.physicalcpu` gives the Performance core count
/// (e.g. 4 on M4 Air). Running decode on P-cores alone typically matches or beats using all
/// P+E cores because P-cores already saturate unified memory bandwidth without E-core stragglers.
pub fn default_perf_threads() -> usize {
    #[cfg(target_os = "macos")]
    {
        if let Some(p_cores) = sysctl_u32(b"hw.perflevel0.physicalcpu\0") {
            if p_cores > 0 {
                return p_cores as usize;
            }
        }
    }
    thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

#[cfg(target_os = "macos")]
fn sysctl_u32(name: &[u8]) -> Option<u32> {
    extern "C" {
        fn sysctlbyname(
            name: *const u8,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *const c_void,
            newlen: usize,
        ) -> i32;
    }
    let mut val: u32 = 0;
    let mut size = std::mem::size_of::<u32>();
    let rc = unsafe {
        sysctlbyname(
            name.as_ptr(),
            (&mut val as *mut u32).cast::<c_void>(),
            &mut size,
            std::ptr::null(),
            0,
        )
    };
    if rc == 0 && size == std::mem::size_of::<u32>() {
        Some(val)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_sum_and_shard_coverage() {
        let pool = ThreadPool::new(4);
        let data: Vec<u64> = (1..=1000).collect();
        let partials = [
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
        ];

        for _ in 0..10 {
            for p in &partials {
                p.store(0, Ordering::Relaxed);
            }
            pool.parallel_for(|tid, n| {
                let range = shard_range(data.len(), tid, n);
                let s: u64 = data[range].iter().sum();
                partials[tid].store(s, Ordering::Relaxed);
            });
            let total: u64 = partials.iter().map(|p| p.load(Ordering::Relaxed)).sum();
            assert_eq!(total, 500_500);
        }
    }
}
