//! Zero-dependency read-only memory mapping and process memory telemetry.
//!
//! Uses direct `extern "C"` bindings to the platform C library (`libSystem` on macOS,
//! `libc` on Linux — already linked by Rust `std`) so GGUF tensor weights stay backed
//! by OS page cache rather than duplicating model files onto the Rust heap.

use std::fs::File;
use std::io;
use std::ops::Deref;
use std::path::Path;
use std::ptr::NonNull;

#[cfg(unix)]
mod sys {
    use std::ffi::c_void;

    pub const PROT_READ: i32 = 1;
    pub const MAP_PRIVATE: i32 = 2;
    pub const MAP_FAILED: *mut c_void = usize::MAX as *mut c_void;

    #[cfg(target_os = "macos")]
    pub const MADV_RANDOM: i32 = 1;
    #[cfg(target_os = "macos")]
    pub const MADV_SEQUENTIAL: i32 = 2;
    #[cfg(target_os = "macos")]
    pub const MADV_WILLNEED: i32 = 3;

    #[cfg(target_os = "linux")]
    pub const MADV_RANDOM: i32 = 1;
    #[cfg(target_os = "linux")]
    pub const MADV_SEQUENTIAL: i32 = 2;
    #[cfg(target_os = "linux")]
    pub const MADV_WILLNEED: i32 = 3;

    extern "C" {
        pub fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            offset: i64,
        ) -> *mut c_void;
        pub fn munmap(addr: *mut c_void, len: usize) -> i32;
        pub fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
        pub fn getrusage(who: i32, usage: *mut u8) -> i32;
    }
}

/// Backing store for model weights: either an OS `mmap` region or an owned buffer.
pub struct MappedFile {
    backing: Backing,
}

enum Backing {
    #[cfg(unix)]
    Mmap { ptr: NonNull<u8>, len: usize },
    Owned(Vec<u8>),
}

// Safety: read-only memory mapping is immutable across all worker threads for its lifetime.
unsafe impl Send for MappedFile {}
unsafe impl Sync for MappedFile {}

impl MappedFile {
    /// Opens `path` as a read-only private memory map on Unix, or reads into memory on non-Unix.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = File::open(path)?;
        let len = file.metadata()?.len() as usize;
        if len == 0 {
            return Ok(Self {
                backing: Backing::Owned(Vec::new()),
            });
        }

        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            // Safety: valid fd and non-zero len; mapping is private and read-only.
            let raw = unsafe {
                sys::mmap(
                    std::ptr::null_mut(),
                    len,
                    sys::PROT_READ,
                    sys::MAP_PRIVATE,
                    file.as_raw_fd(),
                    0,
                )
            };
            if raw == sys::MAP_FAILED {
                return Err(io::Error::last_os_error());
            }
            let ptr = NonNull::new(raw.cast::<u8>())
                .ok_or_else(|| io::Error::other("mmap returned null"))?;
            // Hint sequential read while parsing headers/weights initially.
            unsafe {
                sys::madvise(raw, len, sys::MADV_SEQUENTIAL);
            }
            Ok(Self {
                backing: Backing::Mmap { ptr, len },
            })
        }

        #[cfg(not(unix))]
        {
            use std::io::Read;
            let mut buf = Vec::with_capacity(len);
            let mut f = file;
            f.read_to_end(&mut buf)?;
            Ok(Self {
                backing: Backing::Owned(buf),
            })
        }
    }

    /// Wraps an in-memory byte vector (useful for tests and synthetic GGUF generation).
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self {
            backing: Backing::Owned(bytes),
        }
    }

    /// Hints to the OS kernel to prefetch mapped weight pages into RAM (`MADV_WILLNEED`)
    /// so the first generated token doesn't suffer cold demand-paging latency spikes.
    pub fn advise_willneed(&self) {
        #[cfg(unix)]
        if let Backing::Mmap { ptr, len } = self.backing {
            unsafe {
                sys::madvise(ptr.as_ptr().cast(), len, sys::MADV_WILLNEED);
            }
        }
    }

    /// Hints random access pattern if jumping across unaligned tensor views.
    pub fn advise_random(&self) {
        #[cfg(unix)]
        if let Backing::Mmap { ptr, len } = self.backing {
            unsafe {
                sys::madvise(ptr.as_ptr().cast(), len, sys::MADV_RANDOM);
            }
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        match &self.backing {
            #[cfg(unix)]
            Backing::Mmap { ptr, len } => unsafe { std::slice::from_raw_parts(ptr.as_ptr(), *len) },
            Backing::Owned(vec) => vec.as_slice(),
        }
    }
}

impl Deref for MappedFile {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl Drop for MappedFile {
    fn drop(&mut self) {
        #[cfg(unix)]
        if let Backing::Mmap { ptr, len } = self.backing {
            unsafe {
                sys::munmap(ptr.as_ptr().cast(), len);
            }
        }
    }
}

/// Returns the process Peak Resident Set Size (RSS) in bytes using `getrusage(RUSAGE_SELF)`.
/// Note: macOS reports `ru_maxrss` in bytes; Linux reports it in kilobytes.
pub fn peak_rss_bytes() -> Option<usize> {
    #[cfg(unix)]
    {
        // `struct rusage` starts with two 16-byte `timeval` structs on 64-bit macOS & Linux,
        // placing `ru_maxrss` (`long` / `isize`) at byte offset 32. Total struct size <= 144 bytes.
        let mut buf = [0u8; 256];
        let rc = unsafe { sys::getrusage(0, buf.as_mut_ptr()) };
        if rc != 0 {
            return None;
        }
        let raw = isize::from_ne_bytes(buf[32..32 + std::mem::size_of::<isize>()].try_into().ok()?);
        if raw < 0 {
            return None;
        }
        #[cfg(target_os = "macos")]
        {
            Some(raw as usize)
        }
        #[cfg(target_os = "linux")]
        {
            Some((raw as usize).saturating_mul(1024))
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            None
        }
    }
    #[cfg(not(unix))]
    {
        None
    }
}
