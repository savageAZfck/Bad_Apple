//! Fixed-size, cache-line-aligned memory arena.
//!
//! The arena owns a single contiguous block of memory (`Box<[u8]>`) and hands
//! out slices via a bump pointer.  Resetting the arena only moves the pointer
//! back to the start; it does not call the system allocator and does not drop
//! the contents of previously allocated slices.
//!
//! This is intentionally a low-level tool.  It is **not** a replacement for
//! Candle's internal tensor allocator, but it can eliminate the per-tick
//! `Vec<f64>` allocations that the high-level Rust code creates for embeddings,
//! sensory anchors, and scratch buffers.

use std::alloc::{alloc, handle_alloc_error, Layout};
use std::ptr::NonNull;

/// A 128-byte aligned, fixed-capacity memory arena.
///
/// The 128-byte alignment matches the Apple Silicon high-performance core cache
/// line.  By bump-allocating from a single contiguous block, the main loop
/// avoids repeated `malloc`/`free` calls and the false-sharing that can occur
/// when unrelated vectors land on the same cache line.
#[derive(Debug)]
#[repr(align(128))]
pub struct MemoryArena {
    ptr: NonNull<u8>,
    cap: usize,
    /// Current bump offset in bytes.  Always rounded up to the next 16-byte
    /// boundary so `f32`/`f64` slices are properly aligned.
    offset: usize,
    /// Phantom owner so the arena owns the allocation even though the pointer
    /// is non-HeapSmart.
    _own: Box<[u8]>,
}

impl MemoryArena {
    /// Create a new arena with the requested capacity in bytes.
    ///
    /// The actual allocation is padded to a multiple of 128 bytes so every
    /// slice handed out starts on a 128-byte aligned boundary.
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.next_multiple_of(128).max(128);
        let layout = Layout::from_size_align(cap, 128).expect("arena layout");
        let ptr =
            unsafe { NonNull::new(alloc(layout)).unwrap_or_else(|| handle_alloc_error(layout)) };
        Self {
            ptr,
            cap,
            offset: 0,
            _own: unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr.as_ptr(), cap)) },
        }
    }

    /// Reset the bump pointer without freeing or dropping memory.
    #[inline]
    pub fn reset(&mut self) {
        self.offset = 0;
    }

    /// Allocate an `f32` slice of `len` elements from the arena.
    ///
    /// Returns `None` if the arena does not have enough capacity.  The slice is
    /// guaranteed to be at least 16-byte aligned (4 floats per 128-bit NEON
    /// vector) and, because the arena itself is 128-byte aligned, the slice
    /// starts on a 128-byte cache-line boundary.
    pub fn alloc_f32(&mut self, len: usize) -> Option<&mut [f32]> {
        self.alloc_typed::<f32>(len)
    }

    /// Allocate an `f64` slice of `len` elements from the arena.
    pub fn alloc_f64(&mut self, len: usize) -> Option<&mut [f64]> {
        self.alloc_typed::<f64>(len)
    }

    /// Allocate a byte slice of `len` bytes.
    pub fn alloc_bytes(&mut self, len: usize) -> Option<&mut [u8]> {
        let aligned_len = len.next_multiple_of(16);
        if self.offset + aligned_len > self.cap {
            return None;
        }
        let start = self.ptr.as_ptr().wrapping_add(self.offset);
        self.offset += aligned_len;
        Some(unsafe { std::slice::from_raw_parts_mut(start, len) })
    }

    #[inline]
    fn alloc_typed<T>(&mut self, len: usize) -> Option<&mut [T]> {
        let align = std::mem::align_of::<T>().max(16);
        let size = std::mem::size_of::<T>().saturating_mul(len);
        if size == 0 {
            return Some(&mut []);
        }
        let aligned_size = size.next_multiple_of(align);
        let aligned_offset = self.offset.next_multiple_of(align);
        if aligned_offset + aligned_size > self.cap {
            return None;
        }
        let start = self.ptr.as_ptr().wrapping_add(aligned_offset).cast::<T>();
        self.offset = aligned_offset + aligned_size;
        Some(unsafe { std::slice::from_raw_parts_mut(start, len) })
    }

    /// Current used bytes and total capacity.
    pub fn usage(&self) -> (usize, usize) {
        (self.offset, self.cap)
    }
}

unsafe impl Send for MemoryArena {}
unsafe impl Sync for MemoryArena {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_bump_alloc_and_reset() {
        let mut arena = MemoryArena::new(1024);

        let a = arena.alloc_f64(64).unwrap();
        assert_eq!(a.as_ptr() as usize % 128, 0);
        a.iter_mut().for_each(|v| *v = 1.0);

        let b = arena.alloc_f32(64).unwrap();
        assert_eq!(b.as_ptr() as usize % 16, 0);
        b.iter_mut().for_each(|v| *v = 2.0);

        let (used, _cap) = arena.usage();
        assert!(used > 0);

        arena.reset();
        let (used_after, _cap) = arena.usage();
        assert_eq!(used_after, 0);

        let c = arena.alloc_f64(64).unwrap();
        assert_eq!(c.as_ptr() as usize % 128, 0);
        assert!(arena.alloc_f64(1_000_000).is_none());
    }
}
