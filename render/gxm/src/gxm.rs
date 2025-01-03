use vitasdk_sys::*;
use std::{alloc::Layout, collections::{hash_map::Entry, HashMap}, fmt::Debug, mem::MaybeUninit, ptr::NonNull, rc::Rc, sync::{Mutex, OnceLock}};
use rlsf::Tlsf;
use embedded_graphics::{
    mono_font::{ascii::FONT_10X20, MonoTextStyle}, pixelcolor::Rgb888, prelude::*, primitives::{Circle, PrimitiveStyle}, text::{Alignment, Text}
};

use crate::{sce_err, to_cstring, ALIGN};

const VDM_RING_BUFFER_SIZE: usize = 10 * 1024 * 1024;
const VERTEX_RING_BUFFER_SIZE: usize = SCE_GXM_DEFAULT_VERTEX_RING_BUFFER_SIZE as usize;
const FRAGMENT_RING_BUFFER_SIZE: usize = SCE_GXM_DEFAULT_FRAGMENT_RING_BUFFER_SIZE as usize;
const FRAGMENT_USSE_RING_BUFFER_SIZE: usize = SCE_GXM_DEFAULT_FRAGMENT_USSE_RING_BUFFER_SIZE as usize;

const HEAP_SIZE_LPDDR_R: usize = 64*1024*1024 + VDM_RING_BUFFER_SIZE + VERTEX_RING_BUFFER_SIZE + FRAGMENT_RING_BUFFER_SIZE + FRAGMENT_USSE_RING_BUFFER_SIZE;
const HEAP_SIZE_LPDDR_RW: usize = 32*1024*1024;
const HEAP_SIZE_CDRAM_RW: usize = 32*1024*1024;
const HEAP_SIZE_VERTEX_USSE: usize = 1*1024*1024;
const HEAP_SIZE_FRAGMENT_USSE: usize = 1*1024*1024;

#[allow(non_camel_case_types)]
#[derive(Debug,Clone,Copy)]
pub enum HeapType {
    LPDDR_R,
	LPDDR_RW,
	CDRAM_RW,
	VERTEX_USSE,
	FRAGMENT_USSE
}

#[derive(Debug,Clone,Copy,PartialEq,Eq,Hash)]
pub enum MemoryUsage {
    Generic,
    IndexBuffer,
    VertexBuffer,
    Texture,
    VertexUsse,
    FragmentUsse,
    RingBuffers,
    Display,
}

type Heap = Tlsf<'static, u32, u16, 20, 8>;


#[derive(Debug)]
pub struct Memory {
    heap_lpddr_r: Heap,
    heap_lpddr_rw: Heap,
    heap_cdram_rw: Heap,
    
    heap_vertex_usse: Heap,
    vertex_usse_cpu: usize,
    vertex_usse_offset: usize,
    
    heap_fragment_usse: Heap,
    fragment_usse_cpu: usize,
    fragment_usse_offset: usize,

    // allocations, used
    allocation_sizes: HashMap<usize, (usize, MemoryUsage)>,
    memory_usage: HashMap<MemoryUsage, (usize, usize)>
}

macro_rules! heap_create {
    ($mem:expr, $heap:expr, $size:expr, $attrib:expr) => {{
        sce_err!(sceGxmMapMemory, $mem.as_mut_ptr().cast(), $size as u32, $attrib);
        unsafe {
            $heap.insert_free_block_ptr(NonNull::new(&mut $mem[..$size]).unwrap())
        };
        $mem = &mut $mem[$size..];
    }};
}

macro_rules! heap_usse_create {
    ($mem:expr, $heap:expr, $size:expr, $cpu:expr, $offset:expr, $mapFunc:expr) => {{
        sce_err!($mapFunc,
            $cpu,
            $size as u32,
            &mut $offset
        );
        unsafe {
            $heap.insert_free_block_ptr(
                NonNull::new(&mut $mem[..$size]).unwrap()
            );
        }
        $mem = &mut $mem[$size..];
    }};
}

impl Memory {
    pub fn new() -> Memory {
        const HEAP_SIZE_LPDDR: usize = HEAP_SIZE_LPDDR_R + HEAP_SIZE_LPDDR_RW + HEAP_SIZE_VERTEX_USSE + HEAP_SIZE_FRAGMENT_USSE;

        let mut heap_lpddr_r = Tlsf::new();
        let mut heap_lpddr_rw = Tlsf::new();
        let mut heap_cdram_rw = Tlsf::new();
        let mut heap_vertex_usse = Tlsf::new();
        let mut heap_fragment_usse = Tlsf::new();

        let (lpddr_base, _lpddr_uid) = Memory::alloc_memblock(
            "Lpddr",
            SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE,
            HEAP_SIZE_LPDDR,
        );

        let mut lpddr_mem = unsafe { std::slice::from_raw_parts_mut(
            lpddr_base as *mut u8,
            HEAP_SIZE_LPDDR as usize
        ) };
    
        let (cdram_base, _cdram_uid) = Memory::alloc_memblock(
            "Cdram",
            SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW,
            HEAP_SIZE_CDRAM_RW,
        );

        let mut cdram_mem = unsafe { std::slice::from_raw_parts_mut(
            cdram_base as *mut u8,
            HEAP_SIZE_CDRAM_RW
        ) };

        heap_create!(lpddr_mem, heap_lpddr_r, HEAP_SIZE_LPDDR_R, SCE_GXM_MEMORY_ATTRIB_READ);
        heap_create!(lpddr_mem, heap_lpddr_rw, HEAP_SIZE_LPDDR_RW, SCE_GXM_MEMORY_ATTRIB_READ | SCE_GXM_MEMORY_ATTRIB_WRITE);
        heap_create!(cdram_mem, heap_cdram_rw, HEAP_SIZE_CDRAM_RW, SCE_GXM_MEMORY_ATTRIB_READ | SCE_GXM_MEMORY_ATTRIB_WRITE);

        let mut vertex_usse_offset: u32 = 0;
        let vertex_usse_cpu = lpddr_mem.as_mut_ptr().cast();
        heap_usse_create!(lpddr_mem, heap_vertex_usse, HEAP_SIZE_VERTEX_USSE, vertex_usse_cpu, vertex_usse_offset, sceGxmMapVertexUsseMemory);
        
        let mut fragment_usse_offset: u32 = 0;
        let fragment_usse_cpu = lpddr_mem.as_mut_ptr().cast();
        heap_usse_create!(lpddr_mem, heap_fragment_usse, HEAP_SIZE_FRAGMENT_USSE, fragment_usse_cpu, fragment_usse_offset, sceGxmMapFragmentUsseMemory);

        let _ = lpddr_mem;
        let _ = cdram_mem;

        let allocation_sizes = HashMap::new();
        let mut memory_usage = HashMap::new();
        for usage in [
            MemoryUsage::Generic,
            MemoryUsage::IndexBuffer,
            MemoryUsage::VertexBuffer,
            MemoryUsage::Texture,
            MemoryUsage::VertexUsse,
            MemoryUsage::FragmentUsse,
            MemoryUsage::RingBuffers,
            MemoryUsage::Display,
        ] { memory_usage.insert(usage, (0, 0)); }

        Memory{
            heap_lpddr_r,
            heap_lpddr_rw,
            heap_cdram_rw,
            heap_vertex_usse,
            vertex_usse_cpu: vertex_usse_cpu as usize,
            vertex_usse_offset: vertex_usse_offset as usize,
            heap_fragment_usse,
            fragment_usse_cpu: fragment_usse_cpu as usize,
            fragment_usse_offset: fragment_usse_offset as usize,
            allocation_sizes,
            memory_usage,
        }
    }

    fn alloc_memblock(name: &str, memblock_type: u32, size: usize) -> (*mut c_void, SceUID) {
        let mut buf: *mut c_void = ::core::ptr::null_mut();
        let block_uid = sce_err!(sceKernelAllocMemBlock,
            to_cstring!(name).as_ptr(),
            memblock_type,
            size as u32,
            ::core::ptr::null_mut()
        );
        sce_err!(sceKernelGetMemBlockBase, block_uid, &mut buf);
        (buf, block_uid)
    }

    pub fn allocate<T>(&mut self, heap_type: HeapType, usage: MemoryUsage, size: usize, align: usize) -> *mut T {
        let heap = match heap_type {
            HeapType::LPDDR_R => &mut self.heap_lpddr_r,
            HeapType::LPDDR_RW => &mut self.heap_lpddr_rw,
            HeapType::CDRAM_RW => &mut self.heap_cdram_rw,
            HeapType::FRAGMENT_USSE => &mut self.heap_fragment_usse,
            HeapType::VERTEX_USSE => &mut self.heap_vertex_usse,
        };
        let layout = Layout::from_size_align(size, align).unwrap();
        let ptr = match heap.allocate(layout) {
            Some(v) => v.as_ptr(),
            None => panic!("allocate failed {heap_type:?}")
        };

        self.allocation_sizes.insert(ptr as usize, (size, usage.clone()));

        let (allocations, used) = match self.memory_usage.get(&usage) {
            None => panic!("unhandled memory type {usage:?}"),
            Some(v) => *v
        };
        self.memory_usage.insert(usage, (allocations + 1, used + size));

        println!("allocate size: {} align: {} heap: {:?}", layout.size(), layout.align(), heap_type);
        ptr as *mut T
    }

    pub fn free<T>(&mut self, heap_type: HeapType, ptr: *mut T, align: usize) {
        let heap = match heap_type {
            HeapType::LPDDR_R => &mut self.heap_lpddr_r,
            HeapType::LPDDR_RW => &mut self.heap_lpddr_rw,
            HeapType::CDRAM_RW => &mut self.heap_cdram_rw,
            HeapType::FRAGMENT_USSE => &mut self.heap_fragment_usse,
            HeapType::VERTEX_USSE => &mut self.heap_vertex_usse,
        };

        let (size, usage) = *self.allocation_sizes.get(&(ptr as usize)).unwrap();
        let (allocations, used) = *self.memory_usage.get(&usage).expect("unhandled memory type");
        self.memory_usage.insert(usage, (allocations - 1, used - size));

        unsafe {
            heap.deallocate(NonNull::new_unchecked(ptr as *mut u8), align);
        }
    }

    fn draw_debug_overlay(&mut self, display: &mut DisplayData) {
        let style = MonoTextStyle::new(&FONT_10X20, Rgb888::new(0x00, 0xff, 0x00));
        let mut lines: String = "".to_string();
        for (usage, (allocations, used)) in self.memory_usage.iter() {
            let size = bytesize::ByteSize(*used as u64);
            lines.push_str(&format!("{usage:?} {allocations}, {size}\n"));
        }

        Text::with_alignment(
            &lines,
            Point::new(20, 20),
            style,
            Alignment::Left,
        ).draw(display).unwrap();
    }
}

fn memory() -> &'static Mutex<Memory> {
    static MEMORY: OnceLock<Mutex<Memory>> = OnceLock::new();
    MEMORY.get_or_init(|| Mutex::new(Memory::new()))
}

#[derive(Debug,Clone)]
pub struct Buffer<T> {
    heap_type: HeapType,
    pub ptr: *mut T,
    pub len: usize,
    align: usize,
}

impl<T> Buffer<T> {
    pub fn new(heap_type: HeapType, usage: MemoryUsage, len: usize, align: usize) -> Buffer<T> {
        let mut memory = memory().lock().unwrap();
        let ptr = memory.allocate(heap_type.clone(), usage, len * std::mem::size_of::<T>(), align);
        Buffer {
            heap_type,
            ptr,
            len,
            align
        }
    }

    pub fn from_slice(heap_type: HeapType, usage: MemoryUsage, src: &[T]) -> Buffer<T> {
        let buf = Buffer::new(heap_type, usage, src.len(), 4);
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), buf.ptr, buf.len) };
        buf
    }

    #[allow(unused)]
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len)  }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn copy_from_slice(&mut self, src: &[T]) {
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), self.ptr, self.len) };
    }
}

impl<T> Drop for Buffer<T> {
    fn drop(&mut self) {
        if self.ptr.is_null() { return; }
        let mut memory = memory().lock().unwrap();
        memory.free(self.heap_type, self.ptr, self.align);
    }
}

struct DisplayData {
    address: *mut c_void,
    width: u32,
    height: u32,
    stride: u32,
    flip_mode: u32,
}

impl DrawTarget for DisplayData {
    type Color = Rgb888;
    type Error = u32;

    fn draw_iter<I>(&mut self, pixels: I) -> Result<(), Self::Error>
    where
        I: IntoIterator<Item = Pixel<Self::Color>> {
        for Pixel(coord, color) in pixels.into_iter() {
            let color = color.to_le_bytes();
            unsafe {
                let pixel_ptr = self.address.add(((coord.y * self.stride as i32 + coord.x) * 4) as usize) as *mut u8;
                let pixel = std::ptr::slice_from_raw_parts_mut(pixel_ptr, 4);
                (*pixel)[0] = color[2];
                (*pixel)[1] = color[1];
                (*pixel)[2] = color[0];
                (*pixel)[3] = 0xff;
            }
        }
        Ok(())
    }
}

impl OriginDimensions for DisplayData {
    fn size(&self) -> Size {
        Size::new(self.width, self.height)
    }
}

unsafe extern "C" fn display_callback(callback_data: *const c_void) {
    const SCE_DISPLAY_PIXELFORMAT_A8B8G8R8: u32 = 0;

    let mut display_data = &mut *(callback_data as *mut DisplayData);
    
    let mut memory = memory().lock().unwrap();

    memory.draw_debug_overlay(&mut display_data);

    sceDisplaySetFrameBuf(&SceDisplayFrameBuf{
        size: std::mem::size_of::<SceDisplayFrameBuf>() as u32,
        base: display_data.address,
        pitch: display_data.stride,
        pixelformat: SCE_DISPLAY_PIXELFORMAT_A8B8G8R8,
        width: display_data.width,
        height: display_data.height,
    }, display_data.flip_mode);
    sceDisplayWaitSetFrameBuf();
}


fn init_gxm() {
    let initialize_params = SceGxmInitializeParams{
        flags: 0,
        displayQueueMaxPendingCount: 2,
        displayQueueCallback: Some(display_callback),
        displayQueueCallbackDataSize: std::mem::size_of::<DisplayData>() as u32,
        parameterBufferSize: SCE_GXM_DEFAULT_PARAMETER_BUFFER_SIZE,
    };
    sce_err!(sceGxmInitialize, &initialize_params);
}

pub struct Context {
    pub context: *mut SceGxmContext,
    host_mem: Vec<u8>,
    vdm_mem: Buffer<u8>,
    vertex_mem: Buffer<u8>,
    fragment_mem: Buffer<u8>,
    fragment_usse_mem: Buffer<u8>,
}

impl Drop for Context {
    fn drop(&mut self) {
        sce_err!(sceGxmDestroyContext, self.context);
        let _ = self.host_mem;
        let _ = self.vdm_mem;
        let _ = self.vertex_mem;
        let _ = self.fragment_mem;
        let _ = self.fragment_usse_mem;
    }
}

#[link(name = "SceRazorCapture_stub", kind = "static")]
extern "C" {
    pub fn sceRazorGpuCaptureSetTrigger(frames: u32, path: *const std::ffi::c_char);
    pub fn sceRazorGpuCaptureEnableSalvage(path: *const std::ffi::c_char);
}

fn sce_kernel_load_start_module(name: &str) -> Result<(), &'static str> {
    let mod_id = unsafe { _sceKernelLoadModule(to_cstring!(name).as_ptr(), 0, std::ptr::null()) };
    if mod_id < 0 {
        return Err("Failed to load");
    }
    let mut res: i32 = 0;
    sce_err!(sceKernelStartModule, mod_id, 0, std::ptr::null_mut(), 0, std::ptr::null_mut(), &mut res);
    println!("sce_kernel_load_start_module {name} success!");
    return Ok(());
}

fn init_razor_capture() {
    //sce_err!(sceSysmoduleLoadModule, SCE_SYSMODULE_RAZOR_CAPTURE);
    match sce_kernel_load_start_module("app0:librazorcapture_es4.suprx") {
        Ok(_) => {
            println!("loaded razor, will capture at frame 20");
            unsafe {
                sceRazorGpuCaptureSetTrigger(20, to_cstring!("ux0:data/capture.sgx").as_ptr());
                sceRazorGpuCaptureEnableSalvage(to_cstring!("ux0:data/gpu_crash.sgx").as_ptr());
            }
        }
        Err(e) => {
            println!("not enabling frame capture, razor: {}", e)
        }            
    }
}

impl Context {
    pub fn new() -> Context {
        init_razor_capture();
        init_gxm();

        let mut host_mem: Vec<u8> = Vec::new();
        host_mem.resize(2 * 1024, 0);

        let vdm_mem: Buffer<u8> = Buffer::new(
            HeapType::LPDDR_R,
            MemoryUsage::RingBuffers,
            VDM_RING_BUFFER_SIZE as usize,
            4
        );
        let vertex_mem: Buffer<u8> = Buffer::new(
            HeapType::LPDDR_R,
            MemoryUsage::RingBuffers,
            VERTEX_RING_BUFFER_SIZE as usize,
            4
        );
        let fragment_mem: Buffer<u8> = Buffer::new(
            HeapType::LPDDR_R,
            MemoryUsage::RingBuffers,
            FRAGMENT_RING_BUFFER_SIZE as usize,
            4
        );
        let fragment_usse_mem: Buffer<u8> = Buffer::new(
            HeapType::FRAGMENT_USSE,
            MemoryUsage::RingBuffers,
            FRAGMENT_USSE_RING_BUFFER_SIZE as usize,
            4
        );
        let fragment_usse_offset = unsafe {
            let memory = memory().lock().unwrap();
            fragment_usse_mem.ptr.sub(memory.fragment_usse_cpu as usize).add(memory.fragment_usse_offset as usize) as u32
        };

        let context_params = SceGxmContextParams{
            hostMem: host_mem.as_mut_ptr().cast(),
            hostMemSize: host_mem.len() as u32,
            vdmRingBufferMem: vdm_mem.ptr.cast(),
            vdmRingBufferMemSize: SCE_GXM_DEFAULT_VDM_RING_BUFFER_SIZE,
            vertexRingBufferMem: vertex_mem.ptr.cast(),
            vertexRingBufferMemSize: SCE_GXM_DEFAULT_VERTEX_RING_BUFFER_SIZE,
            fragmentRingBufferMem: fragment_mem.ptr.cast(),
            fragmentRingBufferMemSize: SCE_GXM_DEFAULT_FRAGMENT_RING_BUFFER_SIZE,
            fragmentUsseRingBufferMem: fragment_usse_mem.ptr.cast(),
            fragmentUsseRingBufferMemSize: SCE_GXM_DEFAULT_FRAGMENT_USSE_RING_BUFFER_SIZE,
            fragmentUsseRingBufferOffset: fragment_usse_offset,
        };

        let mut context: *mut SceGxmContext = std::ptr::null_mut();
        sce_err!(sceGxmCreateContext, &context_params, &mut context);

        Context{
            context,
            host_mem,
            vdm_mem,
            vertex_mem,
            fragment_mem,
            fragment_usse_mem,
        }
    }

    pub fn begin_scene(&mut self, display: &mut Display) {
        sce_err!(sceGxmBeginScene,
            self.context,
            0,
            display.main_render_target,
            std::ptr::null(),
            std::ptr::null_mut(),
            display.display_buffer_sync[display.display_back_buffer_index],
            &mut display.display_surface[display.display_back_buffer_index],
            &display.main_depth_surface
        );
    }

    pub fn end_scene(&mut self) {
        sce_err!(sceGxmEndScene,
            self.context,
            std::ptr::null(),
            std::ptr::null()
        );
    }

    pub fn set_vertex_stream<T>(&mut self, verticies: &Buffer<T>) {
        sce_err!(sceGxmSetVertexStream, self.context, 0, verticies.ptr.cast());
    }

    pub fn set_viewport(&mut self, x: i32, y: i32, width: i32, height: i32) {
        unsafe {
            sceGxmSetViewport(
                self.context,
                x as f32, width as f32,
                y as f32, height as f32,
                0., 1.
            );
        }
    }
    
    pub fn set_depth_write_enable(&mut self, enable: bool) {
        unsafe {
            sceGxmSetFrontDepthWriteEnable(self.context, match enable {
                true => SCE_GXM_DEPTH_WRITE_ENABLED,
                false => SCE_GXM_DEPTH_WRITE_DISABLED,
            });
        }
        if !enable {
            self.set_front_depth_func(SCE_GXM_DEPTH_FUNC_ALWAYS);
        }
    }

    pub fn set_front_depth_func(&mut self, depth_func: SceGxmDepthFunc) {
        unsafe { sceGxmSetFrontDepthFunc(self.context, depth_func); }
    }

    pub fn set_front_stencil_func(
        &self,
        func: SceGxmStencilFunc,
        stencil_fail: SceGxmStencilOp,
        depth_fail: SceGxmStencilOp,
        depth_pass: SceGxmStencilOp,
        compare_mask: u8,
        write_mask: u8,
    ) {
        unsafe {
            sceGxmSetFrontStencilFunc(
                self.context,
                func,
                stencil_fail,
                depth_fail,
                depth_pass,
                compare_mask,
                write_mask
            );
        }
    }

    pub fn use_program(&mut self, program: &ShaderProgram) -> UniformBuffers {
        unsafe { sceGxmSetVertexProgram(self.context, program.vertex_shader.program) };
        unsafe { sceGxmSetFragmentProgram(self.context, program.fragment_shader.program) };
        
        let mut vertex_uniform_buffer: *mut c_void = std::ptr::null_mut();
        sce_err!(sceGxmReserveVertexDefaultUniformBuffer, self.context, &mut vertex_uniform_buffer);

        let mut fragment_uniform_buffer: *mut c_void = std::ptr::null_mut();
        sce_err!(sceGxmReserveFragmentDefaultUniformBuffer, self.context, &mut fragment_uniform_buffer);

        (vertex_uniform_buffer, fragment_uniform_buffer)
    }

    pub fn set_texture(&mut self, texture: &Texture) {
        sce_err!(sceGxmSetFragmentTexture, self.context, 0, &texture.texture);
    }

    pub fn draw(
        &mut self,
        prim_type: SceGxmPrimitiveType,
        index_type: SceGxmIndexFormat,
        index_data: &Buffer<u32>,
        index_count: usize,
    ) {
        //println!("sceGxmDraw prim_type: {prim_type} index_type: {index_type} index_data: {:p} index_count: {}", index_data.ptr, index_data.len);
        sce_err!(sceGxmDraw,
            self.context,
            prim_type,
            index_type,
            index_data.ptr.cast(),
            if index_count == 0 { index_data.len } else { index_count } as u32
        );
    }

    pub fn push_marker(&self, name: &'static str) {
        sce_err!(sceGxmPushUserMarker, self.context, to_cstring!(name).as_ptr());
    }

    pub fn pop_marker(&self) {
        sce_err!(sceGxmPopUserMarker, self.context);
    }
}

pub struct Texture {
    texture: SceGxmTexture,
    pub texture_data: Buffer<u8>,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
}

impl Texture {
    pub fn new(width: u32, height: u32, format: SceGxmTextureFormat, data: &[u8]) -> Texture {
        let texture_data: Buffer<u8> = Buffer::new(
            HeapType::LPDDR_R,
            MemoryUsage::Texture,
            data.len(),
            SCE_GXM_TEXTURE_ALIGNMENT as usize
        );
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), texture_data.ptr, data.len()) };

        let mut texture: SceGxmTexture = unsafe { std::mem::zeroed() };
        sce_err!(sceGxmTextureInitLinear,
            &mut texture,
            texture_data.ptr.cast(),
            format,
            width,
            height,
            0
        );

        let stride = unsafe { sceGxmTextureGetStride(&texture) };
        
        Texture{
            texture,
            texture_data,
            width,
            height,
            stride
        }
    }

    pub fn set_filter(&mut self, mag: SceGxmTextureFilter, min: SceGxmTextureFilter) {
        sce_err!(sceGxmTextureSetMagFilter, &mut self.texture, mag);
        sce_err!(sceGxmTextureSetMinFilter, &mut self.texture, min);
    }

    pub fn set_addr_mode(&mut self, mode: SceGxmTextureAddrMode) {
        sce_err!(sceGxmTextureSetUAddrMode, &mut self.texture, mode);
        sce_err!(sceGxmTextureSetVAddrMode, &mut self.texture, mode);
    }
}

impl Debug for Texture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Texture").finish()
    }
}


const DISPLAY_WIDTH: u32 = 960;
const DISPLAY_HEIGHT: u32 = 544;
const DISPLAY_STRIDE_IN_PIXELS: u32 = 1024;
const DISPLAY_BUFFER_COUNT: usize = 3;

pub struct Display {
    pub width: u32,
    pub height: u32,

    // display queue
    display_buffer_data: [Buffer<u32>; DISPLAY_BUFFER_COUNT],
    display_surface: [SceGxmColorSurface; DISPLAY_BUFFER_COUNT],
    display_buffer_sync: [*mut SceGxmSyncObject; DISPLAY_BUFFER_COUNT],
    display_front_buffer_index: usize,
    display_back_buffer_index: usize,

    // depth buffer
    main_depth_buffer_data: Option<Buffer<u8>>,
    main_stencil_buffer_data: Option<Buffer<u8>>,
    main_depth_surface: SceGxmDepthStencilSurface,

    // render target
    main_render_target: *const SceGxmRenderTarget,
}

impl Display {
    pub fn new(msaa_mode: u32, depth_format: SceGxmDepthStencilFormat) -> Display {
        let mut display_buffer_data: [MaybeUninit<Buffer<u32>>; DISPLAY_BUFFER_COUNT] = MaybeUninit::uninit_array();
        let mut display_surface: [SceGxmColorSurface; DISPLAY_BUFFER_COUNT] = unsafe { std::mem::zeroed() };
        let mut display_buffer_sync: [*mut SceGxmSyncObject; DISPLAY_BUFFER_COUNT] = unsafe { std::mem::zeroed() };
        for i in 0..DISPLAY_BUFFER_COUNT {
            let display_buffer: Buffer<u32> = Buffer::new(
                HeapType::CDRAM_RW,
                MemoryUsage::Display,
                (DISPLAY_STRIDE_IN_PIXELS*DISPLAY_HEIGHT) as usize,
                256
            );
            sce_err!(sceGxmColorSurfaceInit,
                &mut display_surface[i],
                SCE_GXM_COLOR_FORMAT_A8B8G8R8,
                SCE_GXM_COLOR_SURFACE_LINEAR,
                if msaa_mode == SCE_GXM_MULTISAMPLE_NONE { SCE_GXM_COLOR_SURFACE_SCALE_NONE } else { SCE_GXM_COLOR_SURFACE_SCALE_MSAA_DOWNSCALE },
                SCE_GXM_OUTPUT_REGISTER_SIZE_32BIT,
                DISPLAY_WIDTH,
                DISPLAY_HEIGHT,
                DISPLAY_STRIDE_IN_PIXELS,
                display_buffer.ptr.cast()
            );
            display_buffer_data[i] = MaybeUninit::new(display_buffer);
            sce_err!(sceGxmSyncObjectCreate, &mut display_buffer_sync[i]);
        }

        let display_buffer_data: [Buffer<u32>; DISPLAY_BUFFER_COUNT] = unsafe { MaybeUninit::array_assume_init(display_buffer_data) };

        // compute depth buffer dimensions
        const ALIGNED_WIDTH: u32 = ALIGN!(DISPLAY_WIDTH, SCE_GXM_TILE_SIZEX);
        const ALIGNED_HEIGHT: u32 = ALIGN!(DISPLAY_HEIGHT, SCE_GXM_TILE_SIZEY);

        let mut sample_count = ALIGNED_WIDTH*ALIGNED_HEIGHT;
        let mut depth_stride_in_samples = ALIGNED_WIDTH;

        if msaa_mode == SCE_GXM_MULTISAMPLE_4X {
            // samples increase in X and Y
            sample_count *= 4;
            depth_stride_in_samples *= 2;
        } else if msaa_mode == SCE_GXM_MULTISAMPLE_2X {
            // samples increase in Y only
            sample_count *= 2;
        }

        // compute depth buffer bytes per sample
        let (depth_bytes_per_sample, stencil_bytes_per_sample) = match depth_format {
            SCE_GXM_DEPTH_STENCIL_FORMAT_DF32 | SCE_GXM_DEPTH_STENCIL_FORMAT_DF32M => (4, 0),
            SCE_GXM_DEPTH_STENCIL_FORMAT_S8 => (0, 1),
            SCE_GXM_DEPTH_STENCIL_FORMAT_DF32_S8 | SCE_GXM_DEPTH_STENCIL_FORMAT_DF32M_S8 => (4, 1),
            SCE_GXM_DEPTH_STENCIL_FORMAT_S8D24 => (4, 1),
            SCE_GXM_DEPTH_STENCIL_FORMAT_D16 => (2, 0),
            _ => (0, 0)
        };

        let main_depth_buffer_data: Option<Buffer<u8>> = if depth_bytes_per_sample == 0 { None } else {
            Some(Buffer::new(
                HeapType::LPDDR_RW,
                MemoryUsage::Display,
                (depth_bytes_per_sample*sample_count) as usize,
                SCE_GXM_DEPTHSTENCIL_SURFACE_ALIGNMENT as usize
            ))
        };

        let main_stencil_buffer_data: Option<Buffer<u8>> = if stencil_bytes_per_sample == 0 { None } else {
            Some(Buffer::new(
                HeapType::LPDDR_RW,
                MemoryUsage::Display,
                (stencil_bytes_per_sample*sample_count) as usize,
                SCE_GXM_DEPTHSTENCIL_SURFACE_ALIGNMENT as usize
            ))
        };

        let mut main_depth_surface: SceGxmDepthStencilSurface = unsafe { std::mem::zeroed() };
        sce_err!(sceGxmDepthStencilSurfaceInit,
            &mut main_depth_surface,
            depth_format,
            SCE_GXM_DEPTH_STENCIL_SURFACE_LINEAR,
            depth_stride_in_samples,
            match main_depth_buffer_data {
                Some(ref data) => data.ptr.cast(),
                None => std::ptr::null_mut()
            },
            match main_stencil_buffer_data {
                Some(ref data) => data.ptr.cast(),
                None => std::ptr::null_mut()
            }
        );

        let display_front_buffer_index = DISPLAY_BUFFER_COUNT-1;
        sce_err!(sceDisplaySetFrameBuf,
            &SceDisplayFrameBuf{
                size: std::mem::size_of::<SceDisplayFrameBuf>() as u32,
                base: display_buffer_data[display_front_buffer_index].ptr.cast(),
                pitch: DISPLAY_STRIDE_IN_PIXELS,
                pixelformat: SCE_DISPLAY_PIXELFORMAT_A8B8G8R8,
                width: DISPLAY_WIDTH,
                height: DISPLAY_HEIGHT,
            },
            SCE_DISPLAY_SETBUF_NEXTFRAME
        );
        sce_err!(sceDisplayWaitSetFrameBuf,);

        let mut main_render_target: *mut SceGxmRenderTarget = std::ptr::null_mut();
        sce_err!(sceGxmCreateRenderTarget,
            &SceGxmRenderTargetParams{
                flags: 0,
                width: DISPLAY_WIDTH as u16,
                height: DISPLAY_HEIGHT as u16,
                scenesPerFrame: 8,
                multisampleMode: msaa_mode as u16,
                multisampleLocations: 0,
                driverMemBlock: -1,
            },
            &mut main_render_target
        );

        Display{
            width: DISPLAY_WIDTH,
            height: DISPLAY_HEIGHT,

            display_buffer_data,
            display_surface,
            display_buffer_sync,
            display_front_buffer_index,
            display_back_buffer_index: 0,

            main_depth_buffer_data,
            main_stencil_buffer_data,
            main_depth_surface,
            main_render_target,
        }
    }

    pub fn swap_buffers(&mut self) {
        let _ = self.main_depth_buffer_data;
        let _ = self.main_stencil_buffer_data;
        sce_err!(sceGxmPadHeartbeat,
            &self.display_surface[self.display_back_buffer_index],
            self.display_buffer_sync[self.display_back_buffer_index]
        );

        let display_data = DisplayData{
            address: self.display_buffer_data[self.display_back_buffer_index].ptr.cast(),
            width: DISPLAY_WIDTH,
            height: DISPLAY_HEIGHT,
            stride: DISPLAY_STRIDE_IN_PIXELS,
            flip_mode: SCE_DISPLAY_SETBUF_NEXTFRAME,
        };
        sce_err!(sceGxmDisplayQueueAddEntry,
			self.display_buffer_sync[self.display_front_buffer_index],
			self.display_buffer_sync[self.display_back_buffer_index],
			(&raw const display_data).cast()
        );

        self.display_front_buffer_index = self.display_back_buffer_index;
        self.display_back_buffer_index = (self.display_back_buffer_index + 1) % DISPLAY_BUFFER_COUNT;
    }
}


pub struct ShaderPatcher {
    shader_patcher: *mut SceGxmShaderPatcher
}

impl ShaderPatcher {
    unsafe extern "C" fn patcher_host_alloc(_user_data: *mut c_void, size: u32) -> *mut c_void {
        let layout = std::alloc::Layout::from_size_align(size as usize + 4, 4).unwrap();
        let mem = std::alloc::alloc(layout);
        *(mem as *mut u32) = size;
        mem.add(4).cast()
    }
    
    unsafe extern "C" fn patcher_host_free(_user_data: *mut c_void, mem: *mut c_void) {
        let real_mem = mem.sub(4);
        let size = *(real_mem as *mut u32);
        let layout = std::alloc::Layout::from_size_align(size as usize + 4, 4).unwrap();
        std::alloc::dealloc(real_mem.cast(), layout);
    }
    
    unsafe extern "C" fn patcher_buffer_alloc(_user_data: *mut c_void, size: u32) -> *mut c_void {
        let mut memory = memory().lock().unwrap();
        memory.allocate(HeapType::LPDDR_RW, MemoryUsage::Generic, size as usize, 4)
    }
    
    unsafe extern "C" fn patcher_buffer_free(_user_data: *mut c_void, mem: *mut c_void) {
        let mut memory = memory().lock().unwrap();
        memory.free(HeapType::LPDDR_RW, mem, 4)
    }
    
    unsafe extern "C" fn patcher_vertex_usse_alloc(_user_data: *mut c_void, size: u32, usse_offset: *mut c_uint) -> *mut c_void {
        let mut memory = memory().lock().unwrap();
        let mem: *mut u8 = memory.allocate(HeapType::VERTEX_USSE, MemoryUsage::VertexUsse, size as usize, 16);
        *usse_offset = mem.sub(memory.vertex_usse_cpu as usize).add(memory.vertex_usse_offset as usize) as u32;
        return mem as *mut c_void
    }
    
    unsafe extern "C" fn patcher_vertex_usse_free(_user_data: *mut c_void, mem: *mut c_void) {
        let mut memory = memory().lock().unwrap();
        memory.free(HeapType::VERTEX_USSE, mem, 16);
    }

    unsafe extern "C" fn patcher_fragment_usse_alloc(_user_data: *mut c_void, size: u32, usse_offset: *mut c_uint) -> *mut c_void {
        let mut memory = memory().lock().unwrap();
        let mem: *mut u8 = memory.allocate(HeapType::FRAGMENT_USSE, MemoryUsage::FragmentUsse, size as usize, 16);
        *usse_offset = mem.sub(memory.fragment_usse_cpu as usize).add(memory.fragment_usse_offset as usize) as u32;
        return mem as *mut c_void
    }
    
    unsafe extern "C" fn patcher_fragment_usse_free(_user_data: *mut c_void, mem: *mut c_void) {
        let mut memory = memory().lock().unwrap();
        memory.free(HeapType::FRAGMENT_USSE, mem, 16);
    }

    pub fn new() -> ShaderPatcher {
        let patcher_params = SceGxmShaderPatcherParams{
            userData: std::ptr::null_mut(),
            hostAllocCallback: Some(ShaderPatcher::patcher_host_alloc),
            hostFreeCallback: Some(ShaderPatcher::patcher_host_free),
            bufferAllocCallback: Some(ShaderPatcher::patcher_buffer_alloc),
            bufferFreeCallback: Some(ShaderPatcher::patcher_buffer_free),
            bufferMem: std::ptr::null_mut(),
            bufferMemSize: 0,
            vertexUsseAllocCallback: Some(ShaderPatcher::patcher_vertex_usse_alloc),
            vertexUsseFreeCallback: Some(ShaderPatcher::patcher_vertex_usse_free),
            vertexUsseMem: std::ptr::null_mut(),
            vertexUsseMemSize: 0,
            vertexUsseOffset: 0,
            fragmentUsseAllocCallback: Some(ShaderPatcher::patcher_fragment_usse_alloc),
            fragmentUsseFreeCallback: Some(ShaderPatcher::patcher_fragment_usse_free),
            fragmentUsseMem: std::ptr::null_mut(),
            fragmentUsseMemSize: 0,
            fragmentUsseOffset: 0,
        };

        let mut shader_patcher: *mut SceGxmShaderPatcher = std::ptr::null_mut();
        sce_err!(sceGxmShaderPatcherCreate, &patcher_params, &mut shader_patcher);
        ShaderPatcher{
            shader_patcher
        }
    }
}

impl Drop for ShaderPatcher {
    fn drop(&mut self) {
        sce_err!(sceGxmShaderPatcherDestroy, self.shader_patcher);
    }
}

#[derive(Debug)]
pub struct ShaderParameter {
    pub parameter: *const SceGxmProgramParameter,
    pub name: String,
    pub category: u32,
    pub component_count: u32,
    pub semantic: SceGxmParameterSemantic,
    pub param_type: SceGxmParameterType,
    pub offset: u32,
}

impl ShaderParameter {
    pub fn from_parameter(
        parameter: *const SceGxmProgramParameter,
    ) -> Self {
        let name_cstr = unsafe { std::ffi::CStr::from_ptr(sceGxmProgramParameterGetName(parameter)) };
        let name = name_cstr.to_string_lossy().into_owned();
        let category = unsafe { sceGxmProgramParameterGetCategory(parameter) };
        let offset = unsafe { sceGxmProgramParameterGetResourceIndex(parameter) };
        let component_count = unsafe { sceGxmProgramParameterGetComponentCount(parameter) };
        let semantic = unsafe { sceGxmProgramParameterGetSemantic(parameter) };
        let param_type = unsafe { sceGxmProgramParameterGetType(parameter) };
        Self {
            parameter,
            name,
            category,
            component_count,
            semantic,
            param_type,
            offset,
        }
    }
}

pub struct VertexShaderProgram {
    name: &'static str,
    shader_patcher: *mut SceGxmShaderPatcher,
    uniforms: Vec<ShaderParameter>,
    uniform_buffer_size: u32,
    program: *mut SceGxmVertexProgram,
    binary: &'static [u8]
}

impl VertexShaderProgram {
    pub fn new(
        name: &'static str,
        shader_patcher: Rc<ShaderPatcher>,
        shader_binary: &'static [u8],
        stride: u16,
        attributes: &[(SceGxmVertexAttribute, &'static str)]
    ) -> VertexShaderProgram {
        let shader_patcher: *mut SceGxmShaderPatcher = shader_patcher.as_ref().shader_patcher;
        let program: *const SceGxmProgram = shader_binary.as_ptr().cast();

        let mut registered: *mut SceGxmRegisteredProgram = std::ptr::null_mut();
        sce_err!(sceGxmShaderPatcherRegisterProgram, shader_patcher, program, &mut registered);

        let parameter_count = unsafe { sceGxmProgramGetParameterCount(program) };

        let uniform_buffer_size = unsafe { sceGxmProgramGetDefaultUniformBufferSize(program) };

        let mut uniforms = Vec::<ShaderParameter>::new();
        let mut vertex_attributes = Vec::<SceGxmVertexAttribute>::new();

        let streams: [SceGxmVertexStream; 1] = [
            SceGxmVertexStream{
                stride,
                indexSource: SCE_GXM_INDEX_SOURCE_INDEX_32BIT as u16,
            }
        ];

        for index in 0..parameter_count {
            let parameter = unsafe { sceGxmProgramGetParameter(program, index) };
            let param = ShaderParameter::from_parameter(parameter);
            match param.category {
                SCE_GXM_PARAMETER_CATEGORY_ATTRIBUTE => {
                    let mut found = false;
                    for (attr, attr_name) in attributes {
                        if param.name == *attr_name {
                            found = true;
                            let mut attr = *attr;
                            attr.regIndex = unsafe { sceGxmProgramParameterGetResourceIndex(param.parameter) } as u16;
                            vertex_attributes.push(attr);
                            break;
                        }
                    }
                    if !found {
                        panic!("attribute in shader not defined in vertex format {}", param.name)
                    }
                },
                SCE_GXM_PARAMETER_CATEGORY_UNIFORM => {
                    println!("uniform: {:#?}", param);
                    uniforms.push(param);
                },
                _ => {
                    panic!("unexpected parameter category {}", param.category);
                }
            }
        }

        let mut vertex_program: *mut SceGxmVertexProgram = std::ptr::null_mut();
        sce_err!(sceGxmShaderPatcherCreateVertexProgram,
            shader_patcher,
            registered,
            vertex_attributes.as_ptr(),
            vertex_attributes.len() as u32,
            streams.as_ptr(),
            streams.len() as u32,
            &mut vertex_program
        );

        VertexShaderProgram{
            name,
            shader_patcher,
            uniforms,
            uniform_buffer_size,
            program: vertex_program,
            binary: shader_binary,
        }
    }
}

impl Drop for VertexShaderProgram {
    fn drop(&mut self) {
        sce_err!(sceGxmShaderPatcherReleaseVertexProgram, self.shader_patcher, self.program);
    }
}

pub struct FragmentShader {
    name: &'static str,
    uniforms: Vec<ShaderParameter>,
    uniform_buffer_size: u32,
    registered: *mut SceGxmRegisteredProgram,
}

impl FragmentShader {
    pub fn new(
        name: &'static str,
        fragment_binary: &'static [u8],
        shader_patcher: Rc<ShaderPatcher>
    ) -> FragmentShader {
        let shader_patcher: *mut SceGxmShaderPatcher = shader_patcher.as_ref().shader_patcher;
        let program: *const SceGxmProgram = fragment_binary.as_ptr().cast();

        let mut registered: *mut SceGxmRegisteredProgram = std::ptr::null_mut();
        sce_err!(sceGxmShaderPatcherRegisterProgram, shader_patcher, fragment_binary.as_ptr().cast(), &mut registered);

        let parameter_count = unsafe { sceGxmProgramGetParameterCount(program) };
        let uniform_buffer_size = unsafe { sceGxmProgramGetDefaultUniformBufferSize(program) };
        let mut uniforms = Vec::<ShaderParameter>::new();

        for index in 0..parameter_count {
            let parameter = unsafe { sceGxmProgramGetParameter(program, index) };
            let param = ShaderParameter::from_parameter(parameter);

            match param.category {
                SCE_GXM_PARAMETER_CATEGORY_UNIFORM => {
                    uniforms.push(param);
                }
                SCE_GXM_PARAMETER_CATEGORY_SAMPLER => {}
                _ => {
                    panic!("Unexpected parameter category {}", param.category);
                }
            }
        }

        FragmentShader {
            name,
            uniforms,
            uniform_buffer_size,
            registered,
        }
    }
}

impl Debug for FragmentShader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FragmentShader").field("name", &self.name.to_string()).finish()
    }
}

// fragment shader + blend info
pub struct FragmentShaderProgram {
    shader: Rc<FragmentShader>,
    shader_patcher: *mut SceGxmShaderPatcher,
    program: *mut SceGxmFragmentProgram,
}

impl FragmentShaderProgram {
    pub fn new(
        shader: Rc<FragmentShader>,
        blend_info: SceGxmBlendInfo,
        shader_patcher: Rc<ShaderPatcher>,
        vertex_binary: &[u8],
        msaa_mode: SceGxmMultisampleMode,
    ) -> FragmentShaderProgram {
        let shader_patcher_ptr: *mut SceGxmShaderPatcher = shader_patcher.as_ref().shader_patcher;

        let mut fragment_program: *mut SceGxmFragmentProgram = std::ptr::null_mut();
        sce_err!(
            sceGxmShaderPatcherCreateFragmentProgram,
            shader_patcher_ptr,
            shader.registered,
            SCE_GXM_OUTPUT_REGISTER_FORMAT_UCHAR4,
            msaa_mode,
            &blend_info,
            vertex_binary.as_ptr().cast(),
            &mut fragment_program
        );

        FragmentShaderProgram{
            shader,
            program: fragment_program,
            shader_patcher: shader_patcher_ptr
        }
    }
}

impl Drop for FragmentShaderProgram {
    fn drop(&mut self) {
        sce_err!(
            sceGxmShaderPatcherReleaseFragmentProgram,
            self.shader_patcher,
            self.program
        );
    }
}

pub struct ShaderProgram {
    name: &'static str,
    vertex_shader: Rc<VertexShaderProgram>,
    fragment_shader: Rc<FragmentShaderProgram>,
    uniform_map: Rc<Vec<Option<(SceGxmProgramType, usize)>>>, // Maps uniform ID to the index in the `uniforms` vector
}

pub type UniformBuffers = (*mut c_void, *mut c_void);

impl ShaderProgram {
    pub fn new(
        name: &'static str,
        vertex_shader: Rc<VertexShaderProgram>,
        fragment_shader: Rc<FragmentShaderProgram>,
        uniform_map: Rc<Vec<Option<(SceGxmProgramType, usize)>>>
    ) -> ShaderProgram {
        ShaderProgram {
            name,
            vertex_shader,
            fragment_shader,
            uniform_map,
        }
    }
    
    pub fn set_uniform(
        &self,
        id: usize,
        values: &[f32],
        uniform_buffers: UniformBuffers
    ) {
        match self.uniform_map.get(id).and_then(|&x| x) {
            Some((location, index)) => {
                let (uniform, buffer) = match location {
                    SCE_GXM_VERTEX_PROGRAM => (&self.vertex_shader.uniforms[index], uniform_buffers.0),
                    SCE_GXM_FRAGMENT_PROGRAM => (&self.fragment_shader.shader.as_ref().uniforms[index], uniform_buffers.1),
                    _ => panic!("uniform_map invalid program type")
                };
                sce_err!(sceGxmSetUniformDataF,
                    buffer,
                    uniform.parameter,
                    0,
                    values.len() as u32,
                    values.as_ptr()
                );
            },
            None => {
                //panic!("Uniform ID '{}' is not registered", id);
            }
        }
    }

    pub fn set_uniform_data<T: Copy>(
        &self,
        id: usize,
        value: &T,
        uniform_buffers: UniformBuffers
    ) {
        match self.uniform_map.get(id).and_then(|&x| x) {
            Some((location, index)) => {
                let (uniform, buffer) = match location {
                    SCE_GXM_VERTEX_PROGRAM => (&self.vertex_shader.uniforms[index], uniform_buffers.0),
                    SCE_GXM_FRAGMENT_PROGRAM => (&self.fragment_shader.shader.as_ref().uniforms[index], uniform_buffers.1),
                    _ => panic!("uniform_map invalid program type")
                };
                let ptr = unsafe { buffer.add(uniform.offset as usize) } as *mut T;
                unsafe { *ptr = *value };
            },
            None => {
                panic!("Uniform ID '{}' is not registered", id);
            }
        }
    }

    #[allow(unused)]
    pub fn dump_uniform_data(&self, uniform_buffers: UniformBuffers) {
        let vertex_buffer = uniform_buffers.0;
        let fragment_buffer = uniform_buffers.1;
        println!("Uniforms:");
        println!("fragment:");
        for uniform in &self.fragment_shader.shader.uniforms {
            ShaderProgram::print_param(uniform, fragment_buffer);
        }
        println!("");
        println!("vertex:");
        for uniform in &self.vertex_shader.uniforms {
            ShaderProgram::print_param(uniform, vertex_buffer);
        }
    }

    fn print_param(uniform: &ShaderParameter, uniform_data: *mut c_void) {
        let uniform_data = unsafe { uniform_data.add(uniform.offset as usize) as *mut u8 };

        // Print the uniform's details
        println!("Uniform Name: {}", uniform.name);

        macro_rules! print_array {
            ($data:expr, $type:tt, $component_size:expr) => {
                print!("Values: [");
                for i in 0..uniform.component_count {
                    let data = unsafe { *($data.add((i * $component_size) as usize) as *const $type) };
                    if i > 0 {
                        print!(", ");
                    }
                    print!("{:?}", data);
                }
                println!("]");
            };
        }

        // Use the macro to print the values based on type
        match uniform.param_type {
            SCE_GXM_PARAMETER_TYPE_F32 => {
                print_array!(uniform_data, f32, 4);
            }
            SCE_GXM_PARAMETER_TYPE_U32 => {
                print_array!(uniform_data, u32, 4);
            }
            SCE_GXM_PARAMETER_TYPE_S32 => {
                print_array!(uniform_data, i32, 4);
            }
            SCE_GXM_PARAMETER_TYPE_F16 => {
                print_array!(uniform_data, f32, 2);
            }
            SCE_GXM_PARAMETER_TYPE_U16 => {
                print_array!(uniform_data, u16, 2);
            }
            SCE_GXM_PARAMETER_TYPE_S16 => {
                print_array!(uniform_data, i16, 2);
            }
            SCE_GXM_PARAMETER_TYPE_U8 => {
                print_array!(uniform_data, u8, 1);
            }
            SCE_GXM_PARAMETER_TYPE_S8 => {
                print_array!(uniform_data, i8, 1);
            }
            _ => {
                panic!("Unsupported parameter type: {:?}", uniform.param_type);
            }
        }
    }
}

impl Debug for ShaderProgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShaderProgram").field("name", &self.name.to_string()).finish()
    }
}

type BlendState = (u8, u8, u8);

pub struct ShaderRegistry {
    vertex_programs: Vec<Rc<VertexShaderProgram>>,
    fragment_shaders: Vec<Rc<FragmentShader>>,
    combined_shaders: Vec<(usize, usize, Rc<Vec<Option<(SceGxmProgramType, usize)>>>)>,
    shader_programs: HashMap<(usize, BlendState), Rc<ShaderProgram>>,
    uniform_names: &'static [&'static str],
    shader_patcher: Rc<ShaderPatcher>,
    msaa_mode: SceGxmMultisampleMode,
}

impl ShaderRegistry {
    pub fn new(
        uniform_names: &'static [&'static str],
        shader_patcher: ShaderPatcher,
        msaa_mode: SceGxmMultisampleMode,
    ) -> Self {
        Self {
            vertex_programs: Vec::new(),
            fragment_shaders: Vec::new(),
            combined_shaders: Vec::new(),
            shader_programs: HashMap::new(),
            uniform_names,
            shader_patcher: Rc::new(shader_patcher),
            msaa_mode,
        }
    }

    pub fn register_vertex_shader<T: Sized>(
        &mut self,
        name: &'static str,
        vertex_shader_binary: &'static [u8],
        attributes: &[(SceGxmVertexAttribute, &'static str)]
    ) -> Result<usize, String> {
        let vertex_program = VertexShaderProgram::new(
            name,
            Rc::clone(&self.shader_patcher),
            vertex_shader_binary,
            std::mem::size_of::<T>() as u16,
            attributes
        );
        self.vertex_programs.push(Rc::new(vertex_program));
        Ok(self.vertex_programs.len() - 1)
    }

    pub fn register_fragment_shader(
        &mut self,
        name: &'static str,
        fragment_shader_binary: &'static [u8],
    ) -> Result<usize, String> {
        let fragment_shader = FragmentShader::new(
            name,
            fragment_shader_binary,
            self.shader_patcher.clone(),
        );
        self.fragment_shaders.push(Rc::new(fragment_shader));
        Ok(self.fragment_shaders.len() - 1)
    }

    pub fn register_shader(
        &mut self,
        vertex_id: usize,
        fragment_id: usize,
    ) -> Result<usize, String> {
        if vertex_id >= self.vertex_programs.len() || fragment_id >= self.fragment_shaders.len() {
            return Err("Invalid shader ID provided".to_string());
        }

        let vertex_program = &self.vertex_programs[vertex_id];
        let fragment_shader = &self.fragment_shaders[fragment_id];

        let mut uniform_map = Vec::new();
        uniform_map.reserve_exact(self.uniform_names.len());

        for &uniform_name in self.uniform_names.iter() {
            let vertex_index = vertex_program.uniforms.iter().position(|u| u.name == uniform_name);
            let fragment_index = fragment_shader.uniforms.iter().position(|u| u.name == uniform_name);

            let uniform_entry = match (vertex_index, fragment_index) {
                (Some(index), None) => Some((SCE_GXM_VERTEX_PROGRAM, index)),
                (None, Some(index)) => Some((SCE_GXM_FRAGMENT_PROGRAM, index)),
                (Some(_), Some(_)) => {
                    return Err(format!(
                        "Uniform '{}' appears in both vertex and fragment shaders: {}",
                        uniform_name, vertex_program.name
                    ));
                }
                (None, None) => None,
            };
            uniform_map.push(uniform_entry);
        }

        self.combined_shaders.push((vertex_id, fragment_id, Rc::new(uniform_map)));
        Ok(self.combined_shaders.len() - 1)
    }

    pub fn get_shader(
        &mut self,
        shader_id: usize,
        blend_state: BlendState,
    ) -> Result<Rc<ShaderProgram>, String> {
        match self.shader_programs.entry((shader_id, blend_state)) {
            Entry::Occupied(o) => {
                Ok(Rc::clone(o.get()))
            },
            Entry::Vacant(v) => {
                let (vertex_id, fragment_id, uniforms) = self
                    .combined_shaders
                    .get(shader_id)
                    .ok_or_else(|| format!("Unknown shader ID requested: {}", shader_id))?;

                let vertex_program = &self.vertex_programs[*vertex_id];
                let fragment_shader = &self.fragment_shaders[*fragment_id];

                println!("creating new shader {}", vertex_program.name);

                let fragment_program = FragmentShaderProgram::new(
                    Rc::clone(fragment_shader),
                    Self::blend_state_as_gxm(blend_state),
                    Rc::clone(&self.shader_patcher),
                    vertex_program.binary,
                    self.msaa_mode,
                );
        
                let new_program = Rc::new(ShaderProgram::new(
                    vertex_program.name,
                    Rc::clone(vertex_program),
                    Rc::new(fragment_program),
                    Rc::clone(uniforms),
                ));
    
                Ok(v.insert(new_program).clone())
            }
        }
    }

    fn blend_state_as_gxm(blend_state: BlendState) -> SceGxmBlendInfo {
        let mut blend_info: SceGxmBlendInfo = unsafe { std::mem::zeroed() };
        let (blend_op, src_rgb, dst_rgb) = blend_state;
        blend_info.colorMask = SCE_GXM_COLOR_MASK_ALL as u8;
        blend_info.set_colorFunc(blend_op);
        blend_info.set_alphaFunc(SCE_GXM_BLEND_FUNC_ADD as u8);
        blend_info.set_colorSrc(src_rgb);
        blend_info.set_colorDst(dst_rgb);
        blend_info.set_alphaSrc(SCE_GXM_BLEND_FACTOR_ONE as u8);
        blend_info.set_alphaDst(SCE_GXM_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA as u8);
        blend_info
    }
}
