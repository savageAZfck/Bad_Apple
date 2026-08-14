//! Apple Silicon Unified Memory Architecture (UMA) buffer manager.
//!
//! Allocates `MTLBuffer` objects with `MTLResourceStorageModeShared` so the
//! same physical memory is mapped into both the CPU and the GPU address
//! spaces.  This is the foundation for zero-copy tensor/connectome storage on
//! Apple Silicon: the CPU can read and write through `Buffer::contents()` while
//! Metal compute kernels can operate on the same allocation.

use candle_core::backend::BackendStorage;
use candle_core::{DType, Device as CandleDevice, MetalStorage, Storage, Tensor};
use metal::{Buffer, Device as MetalDevice, MTLResourceOptions};
use safetensors::tensor::{serialize_to_file, Dtype, View};
use std::borrow::Cow;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::path::Path;

/// A typed CPU/GPU shared buffer backed by a `MTLBuffer`.
///
/// The backing allocation has `MTLResourceStorageModeShared`, which on Apple
/// Silicon means both host and device can access the same physical pages.
/// Synchronization is the caller's responsibility: either use a single
/// producer/consumer pattern or insert command-encoder barriers/fences when
/// the GPU is involved.
pub struct UmaBuffer<T: Copy + Send + Sync + 'static> {
    buffer: Buffer,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T: Copy + Send + Sync + 'static> UmaBuffer<T> {
    /// Allocate a new shared buffer on the default Metal device.
    ///
    /// Returns `None` if no Metal device is available or if the allocation
    /// fails, so callers can fall back to an ordinary `Vec`.
    pub fn new(len: usize) -> Option<Self> {
        let device = MetalDevice::system_default()?;
        let bytes = len.checked_mul(std::mem::size_of::<T>())? as u64;
        // StorageModeShared is the UMA path on Apple Silicon: both CPU and GPU
        // access the same physical memory without copies.
        let options = MTLResourceOptions::StorageModeShared;
        let buffer = device.new_buffer(bytes, options);
        Some(Self {
            buffer,
            len,
            _marker: PhantomData,
        })
    }

    /// Length of the buffer in elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Untyped access to the underlying `MTLBuffer` for compute encoders.
    pub fn metal_buffer(&self) -> &Buffer {
        &self.buffer
    }

    fn ptr(&self) -> *mut T {
        self.buffer.contents() as *mut T
    }

    /// Return a shared slice into the buffer.
    ///
    /// # Safety
    /// The buffer contents are only valid for as long as the `UmaBuffer` is
    /// alive.  Concurrent GPU access to the same range without a barrier is
    /// undefined behaviour.
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: `new_buffer` returns a length-aligned allocation and the
        // `UmaBuffer` owns it.  The pointer is non-null and points to `len`
        // valid elements of `T` until the `Buffer` is dropped.
        unsafe { std::slice::from_raw_parts(self.ptr(), self.len) }
    }

    /// Return a mutable slice into the buffer.
    ///
    /// # Safety
    /// Same as `as_slice`, and the caller must ensure the GPU is not
    /// reading/writing the same range.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: as `as_slice`, plus the unique `&mut self` borrow guarantees
        // no other Rust reference is active.
        unsafe { std::slice::from_raw_parts_mut(self.ptr(), self.len) }
    }
}

impl<T: Copy + Send + Sync + 'static> Deref for UmaBuffer<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T: Copy + Send + Sync + 'static> DerefMut for UmaBuffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

/// Zero-copy `safetensors::View` that reads directly from a `MTLResourceStorageModeShared`
/// `MetalStorage` buffer.  Because the CPU and GPU already share the same
/// physical pages, the safetensors serializer can write the file straight from
/// the `MTLBuffer` without a Metal -> CPU `to_vec1` copy.
struct UmaSafetensorView {
    storage: MetalStorage,
    dtype: Dtype,
    shape: Vec<usize>,
    /// Byte offset into the `MTLBuffer` where the tensor data begins.
    offset: usize,
    /// Number of bytes to serialize.
    len: usize,
}

impl View for UmaSafetensorView {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        let ptr = self.storage.buffer().contents() as *const u8;
        assert!(
            !ptr.is_null(),
            "UmaSafetensorView::data called on a private/non-readable MTLBuffer"
        );
        // macOS may not wire every page of a new shared MTLBuffer into the
        // CPU address space until it is touched.  A single `read_volatile` per
        // page forces the driver to fault the page in, so the kernel's
        // `copyin` in `std::io::write` can read the full buffer without
        // expensive soft-fault handling.
        const PAGE: usize = 4096;
        let mut i = 0;
        while i < self.len {
            unsafe { std::ptr::read_volatile(ptr.add(self.offset + i)) };
            i += PAGE;
        }
        // SAFETY: `MetalStorage` owns the `Arc<Buffer>`, so the `MTLBuffer` is
        // alive for the lifetime of this view.  The `View` borrow is scoped to
        // the safetensors serialization call, which completes before the view
        // is dropped.
        let slice = unsafe { std::slice::from_raw_parts(ptr.add(self.offset), self.len) };
        Cow::Borrowed(slice)
    }

    fn data_len(&self) -> usize {
        self.len
    }
}

/// Save a map of Candle tensors to a safetensors file using the UMA shared
/// buffer pointer whenever possible.  Metal tensors that are contiguous and
/// backed by `StorageModeShared` are written without an intermediate CPU copy;
/// everything else falls back to the ordinary `candle_core::safetensors::save`
/// path.
/// Make a contiguous Metal tensor readable from the CPU by blitting it to a
/// `MTLResourceStorageModeShared` staging buffer.  Returns a `MetalStorage`
/// that owns the staging buffer; the CPU can read `staging.buffer().contents()`
/// after `wait_until_completed()`.
fn blit_to_shared(
    src: &MetalStorage,
    start: usize,
    len: usize,
    elem_count: usize,
    dtype: DType,
) -> candle_core::Result<MetalStorage> {
    let device = src.device();
    let shared = device.allocate_buffer(len)?;
    let blit = device.blit_command_encoder()?;
    blit.copy_from_buffer(src.buffer(), start, &shared, 0, len);
    blit.end_encoding();
    device.wait_until_completed()?;
    Ok(MetalStorage::new(shared, device.clone(), elem_count, dtype))
}

/// Save a map of Candle tensors to a safetensors file using UMA shared buffers.
///
/// Metal tensors that are contiguous are blitted to a `StorageModeShared` staging
/// buffer and then written directly from the `MTLBuffer` without an
/// intermediate CPU `Vec` copy.  CPU or non-contiguous tensors fall back to
/// Candle's ordinary `safetensors::save`.
pub fn save_metal_tensors<P: AsRef<Path>>(
    tensors: &HashMap<String, Tensor>,
    filename: P,
) -> candle_core::Result<()> {
    let mut views: Vec<(String, UmaSafetensorView)> = Vec::with_capacity(tensors.len());
    let mut fallback_names: Vec<String> = Vec::new();

    for (name, t) in tensors {
        if t.device().is_metal() {
            let (storage, layout) = t.storage_and_layout();
            if let Storage::Metal(m) = &*storage {
                let elem_count = t.dims().iter().product::<usize>();
                let bytes_per = t.dtype().size_in_bytes();
                let start = layout.start_offset() * bytes_per;
                let len = elem_count * bytes_per;
                let buffer_len = m.buffer().length();

                if t.is_contiguous() && start + len <= buffer_len {
                    // Candle allocates most compute buffers as
                    // `StorageModePrivate` on macOS, and even `StorageModeShared`
                    // pages may not be wired for `write(2)`.  Blit the tensor
                    // into a fresh shared staging buffer and serialize from
                    // there.
                    let staging = blit_to_shared(m, start, len, elem_count, t.dtype())?;

                    views.push((
                        name.clone(),
                        UmaSafetensorView {
                            storage: staging,
                            dtype: safetensors_dtype_from_candle(t.dtype()),
                            shape: t.dims().to_vec(),
                            offset: 0,
                            len,
                        },
                    ));
                    continue;
                }
            }
        }
        fallback_names.push(name.clone());
    }

    if fallback_names.is_empty() {
        // Synchronize the GPU before the CPU reads any shared buffer.  Blitted
        // views already waited, but `wait_until_completed` is idempotent.
        for (_, view) in &views {
            view.storage.device().wait_until_completed()?;
        }
        serialize_to_file(views, None, filename.as_ref())
            .map_err(|e| candle_core::Error::Msg(format!("safetensors: {e}")))?;
        Ok(())
    } else {
        // Mixed path: write the UMA tensors directly, then load them back and
        // merge with the fallback tensors before using Candle's `save`.
        let mut merged: HashMap<String, Tensor> = HashMap::with_capacity(tensors.len());

        let tmp = std::env::temp_dir().join(format!(
            "firefly_uma_save_{}.safetensors",
            rand::random::<u64>()
        ));
        for (_, view) in &views {
            view.storage.device().wait_until_completed()?;
        }
        serialize_to_file(views, None, &tmp)
            .map_err(|e| candle_core::Error::Msg(format!("safetensors: {e}")))?;
        let loaded = candle_core::safetensors::load(&tmp, &CandleDevice::Cpu)?;
        let _ = std::fs::remove_file(&tmp);

        for (name, t) in &loaded {
            merged.insert(name.clone(), t.clone());
        }
        for name in fallback_names {
            if let Some(t) = tensors.get(&name) {
                let t = t.to_device(&CandleDevice::Cpu)?;
                merged.insert(name, t);
            }
        }

        candle_core::safetensors::save(&merged, filename)?;
        Ok(())
    }
}

fn safetensors_dtype_from_candle(value: DType) -> Dtype {
    match value {
        DType::U8 => Dtype::U8,
        DType::U32 => Dtype::U32,
        DType::I16 => Dtype::I16,
        DType::I32 => Dtype::I32,
        DType::I64 => Dtype::I64,
        DType::BF16 => Dtype::BF16,
        DType::F16 => Dtype::F16,
        DType::F32 => Dtype::F32,
        DType::F64 => Dtype::F64,
        DType::F8E4M3 => Dtype::F8_E4M3,
        DType::F6E2M3 => Dtype::F6_E2M3,
        DType::F6E3M2 => Dtype::F6_E3M2,
        DType::F4 => Dtype::F4,
        DType::F8E8M0 => Dtype::F8_E8M0,
        _ => Dtype::F32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn uma_buffer_round_trip() {
        let mut buf = UmaBuffer::<f32>::new(16).expect("Metal device needed for this test");
        for (i, x) in buf.as_mut_slice().iter_mut().enumerate() {
            *x = i as f32;
        }
        for (i, x) in buf.as_slice().iter().enumerate() {
            assert_eq!(*x, i as f32);
        }
    }

    /// Measure how long it takes to write a connectome-sized UMA buffer from
    /// the CPU.  On UMA this is a host memory write and requires no PCIe copy.
    #[test]
    fn uma_connectome_write_latency() {
        const ELEM_COUNT: usize = 2048 * 500; // 500 connectome nodes worth of f64
        let mut buf = UmaBuffer::<f64>::new(ELEM_COUNT).expect("Metal device needed for this test");
        let slice = buf.as_mut_slice();

        let samples: Vec<_> = (0..20)
            .map(|_| {
                let start = Instant::now();
                for (i, x) in slice.iter_mut().enumerate() {
                    *x = (i % 1024) as f64;
                }
                start.elapsed().as_micros() as u64
            })
            .collect();

        let avg = samples.iter().sum::<u64>() / samples.len() as u64;
        let min = *samples.iter().min().unwrap();
        let max = *samples.iter().max().unwrap();
        eprintln!(
            "[uma_connectome_write] avg={} µs min={} µs max={} µs (samples: {:?})",
            avg, min, max, samples
        );

        // A 500 * 2048 f64 write should complete in well under a millisecond on
        // UMA (the allocation is ~8 MiB).  Keep a generous ceiling for CI.
        assert!(avg < 2_000, "UMA connectome write averaged {} µs", avg);
    }

    /// Round-trip a small Metal tensor through the UMA safetensors save path.
    #[test]
    fn uma_safetensors_round_trip() {
        use candle_core::Device;

        let device = Device::new_metal(0).unwrap_or(Device::Cpu);
        let t = Tensor::arange(0.0_f32, 16.0_f32, &device).unwrap();
        let mut map = std::collections::HashMap::new();
        map.insert("test".to_string(), t.clone());

        let tmp = std::env::temp_dir().join(format!(
            "firefly_uma_roundtrip_{}.safetensors",
            rand::random::<u64>()
        ));
        save_metal_tensors(&map, &tmp).unwrap();

        let loaded = candle_core::safetensors::load(&tmp, &Device::Cpu).unwrap();
        let actual = &loaded["test"];
        let expected = t.to_device(&Device::Cpu).unwrap();
        assert_eq!(actual.dims(), expected.dims());
        assert!(
            (actual.to_vec1::<f32>().unwrap()[0] - expected.to_vec1::<f32>().unwrap()[0]).abs()
                < 1e-4
        );
    }

    /// Round-trip multiple Metal tensors with mixed 1D/2D shapes.
    #[test]
    fn uma_safetensors_multi_tensor_round_trip() {
        use candle_core::Device;

        let device = Device::new_metal(0).unwrap_or(Device::Cpu);
        let a = Tensor::randn(0.0_f32, 1.0, (576, 2048), &device).unwrap();
        let b = Tensor::randn(0.0_f32, 1.0, 576, &device).unwrap();
        let c = Tensor::randn(0.0_f32, 1.0, (1, 2048), &device).unwrap();
        let mut map = std::collections::HashMap::new();
        map.insert("a".to_string(), a.clone());
        map.insert("b".to_string(), b.clone());
        map.insert("c".to_string(), c.clone());

        let tmp = std::env::temp_dir().join(format!(
            "firefly_uma_multi_{}.safetensors",
            rand::random::<u64>()
        ));
        save_metal_tensors(&map, &tmp).unwrap();

        let loaded = candle_core::safetensors::load(&tmp, &Device::Cpu).unwrap();
        for (name, key) in [("a", "a"), ("b", "b"), ("c", "c")] {
            let actual = &loaded[name];
            let expected = map[key].to_device(&Device::Cpu).unwrap();
            assert_eq!(actual.dims(), expected.dims());
            let a0 = actual.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
            let e0 = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
            assert!(
                (a0 - e0).abs() < 1e-3,
                "tensor {} mismatch: {} vs {}",
                name,
                a0,
                e0
            );
        }
    }

    /// Allocate and compare against an ordinary `Vec<f64>` to make sure the
    /// UMA path does not introduce unexpected overhead on the CPU side.
    #[test]
    fn uma_vs_vec_copy_latency() {
        const ELEM_COUNT: usize = 2048 * 500;
        let mut buf = UmaBuffer::<f64>::new(ELEM_COUNT).expect("Metal device needed for this test");
        let mut vec = vec![0.0_f64; ELEM_COUNT];
        let src = vec![1.0_f64; ELEM_COUNT];

        let uma_us = {
            let start = Instant::now();
            buf.as_mut_slice().copy_from_slice(&src);
            start.elapsed().as_micros() as u64
        };

        let vec_us = {
            let start = Instant::now();
            vec.copy_from_slice(&src);
            start.elapsed().as_micros() as u64
        };

        eprintln!("[uma_vs_vec] uma={} µs vec={} µs", uma_us, vec_us);

        assert_eq!(buf.as_slice(), &src[..]);
        assert_eq!(&vec[..], &src[..]);

        // The UMA copy should be in the same ballpark as a normal `memcpy`.
        // Give it a 3x ceiling to account for first-touch / page-fault noise.
        assert!(
            uma_us < vec_us * 3,
            "UMA copy {} µs was more than 3x slower than Vec copy {} µs",
            uma_us,
            vec_us
        );
    }
}
