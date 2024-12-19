#![deny(clippy::unwrap_used)]
// Remove this when we start using `Rc` when compiling for wasm
#![allow(clippy::arc_with_non_send_sync)]
#![feature(iter_collect_into)]

use bytemuck::{Pod, Zeroable};
use ruffle_render::backend::{
    BitmapCacheEntry, Context3D, Context3DProfile, PixelBenderOutput, PixelBenderTarget,
    RenderBackend, ShapeHandle, ShapeHandleImpl, ViewportDimensions,
};
use ruffle_render::bitmap::{
    Bitmap, BitmapFormat, BitmapHandle, BitmapHandleImpl, BitmapSource, PixelRegion, PixelSnapping,
    RgbaBufRead, SyncHandle,
};
use ruffle_render::commands::{CommandHandler, CommandList, RenderBlendMode};
use ruffle_render::error::Error as BitmapError;
use ruffle_render::matrix::Matrix;
use ruffle_render::quality::StageQuality;
use ruffle_render::shape_utils::DistilledShape;
use ruffle_render::tessellator::{
    Gradient as TessGradient, ShapeTessellator, Vertex as TessVertex,
};
use ruffle_render::transform::Transform;
use std::borrow::Cow;
use std::rc::Rc;
use std::sync::Arc;
use swf::{BlendMode, Color, Twips};
use thiserror::Error;

use vitasdk_sys::*;

mod gxm;
mod macros;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Couldn't create GL context")]
    CantCreateGLContext,

    #[error("Couldn't create frame buffer")]
    UnableToCreateFrameBuffer,

    #[error("Couldn't create program")]
    UnableToCreateProgram,

    #[error("Couldn't create texture")]
    UnableToCreateTexture,

    #[error("Couldn't compile shader")]
    UnableToCreateShader,

    #[error("Couldn't create render buffer")]
    UnableToCreateRenderBuffer,

    #[error("Couldn't create vertex array object")]
    UnableToCreateVAO,

    #[error("Couldn't create buffer")]
    UnableToCreateBuffer,

    #[error("OES_element_index_uint extension not available")]
    OESExtensionNotFound,

    #[error("VAO extension not found")]
    VAOExtensionNotFound,

    #[error("Couldn't link shader program: {0}")]
    LinkingShaderProgram(String),

    #[error("GL Error in {0}: {1}")]
    GLError(&'static str, u32),
}

const COLOR_VERTEX_GXP: &[u8] = include_bytes_align_as!(u32, "../shaders/compiled/color_v.gxp");
const COLOR_FRAGMENT_GXP: &[u8] = include_bytes_align_as!(u32, "../shaders/compiled/color_f.gxp");
const TEXTURE_VERTEX_GXP: &[u8] = include_bytes_align_as!(u32, "../shaders/compiled/texture_v.gxp");
const GRADIENT_FRAGMENT_GXP: &[u8] = include_bytes_align_as!(u32, "../shaders/compiled/gradient_f.gxp");
const BITMAP_FRAGMENT_GXP: &[u8] = include_bytes_align_as!(u32, "../shaders/compiled/bitmap_f.gxp");
const CLEAR_VERTEX_GXP: &[u8] = include_bytes_align_as!(u32, "../shaders/compiled/clear_v.gxp");
const CLEAR_FRAGMENT_GXP: &[u8] = include_bytes_align_as!(u32, "../shaders/compiled/clear_f.gxp");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MaskState {
    NoMask,
    DrawMaskStencil,
    DrawMaskedContent,
    ClearMaskStencil,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Vertex {
    position: [f32; 2],
    color: u32,
}

impl From<TessVertex> for Vertex {
    fn from(vertex: TessVertex) -> Self {
        Self {
            position: [vertex.x, vertex.y],
            color: u32::from_le_bytes([
                vertex.color.r,
                vertex.color.g,
                vertex.color.b,
                vertex.color.a,
            ]),
        }
    }
}


pub struct GxmRenderBackend {
    // gxm context
    context: gxm::Context,
    display: gxm::Display,
    shader_registry: gxm::ShaderRegistry,

    color_shader: usize,
    bitmap_shader: usize,
    gradient_shader: usize,
    clear_shader: usize,

    // The frame buffers used for resolving MSAA.
    msaa_sample_count: u32,

    shape_tessellator: ShapeTessellator,

    quad_verticies: gxm::Buffer<Vertex>,
    quad_indicies: gxm::Buffer<u32>,

    mask_state: MaskState,
    num_masks: u32,
    mask_state_dirty: bool,
    is_transparent: bool,
    blend_state: (u8, u8, u8),
    viewport: ViewportDimensions,
    viewport_dirty: bool,

    blend_modes: Vec<RenderBlendMode>,

    renderbuffer_width: i32,
    renderbuffer_height: i32,
    view_matrix: [[f32; 4]; 4],

    // This is currently unused - we just hold on to it
    // to expose via `get_viewport_dimensions`
    viewport_scale_factor: f64,
}

#[derive(Debug)]
struct RegistryData {
    width: u32,
    height: u32,
    texture: gxm::Texture,
}

impl Drop for RegistryData {
    fn drop(&mut self) {
    }
}

impl BitmapHandleImpl for RegistryData {}

fn as_registry_data(handle: &BitmapHandle) -> &mut RegistryData {
    // needed because RegistryData
    // contains gxm::Texture, which directly has the slice that is the texture in memory,
    // to update the texture it is required to have a mutable reference, nothing else mutates it so it should be fine (tm)
    let ptr = Arc::as_ptr(&handle.0) as *const dyn BitmapHandleImpl;
    let registry_data = unsafe {
        <dyn BitmapHandleImpl>::downcast_mut(&mut *(ptr as *mut dyn BitmapHandleImpl))
    };
    registry_data.expect("Bitmap handle must be gxm RegistryData")
}


impl GxmRenderBackend {
    pub fn new(
        is_transparent: bool,
        quality: StageQuality,
    ) -> Result<Self, Error> {
        // Determine MSAA sample count.
        let msaa_sample_count = quality.sample_count().min(4);

        let depth_format: SceGxmDepthStencilFormat = SCE_GXM_DEPTH_STENCIL_FORMAT_DF32_S8;
        let msaa_mode: SceGxmMultisampleMode = SCE_GXM_MULTISAMPLE_NONE;

        let context = gxm::Context::new();
        let display = gxm::Display::new(msaa_mode, depth_format);
        let shader_patcher = gxm::ShaderPatcher::new();
        let mut shader_registry = gxm::ShaderRegistry::new(&UNIFORM_NAMES, shader_patcher);

        let vertex_stride = std::mem::size_of::<Vertex>() as u16;
        let vertex_attributes: [(SceGxmVertexAttribute, &'static str); 2] = [
            (SceGxmVertexAttribute{
                streamIndex: 0,
                offset: 0,
                format: SCE_GXM_ATTRIBUTE_FORMAT_F32 as u8,
                componentCount: 2,
                regIndex: 0,
            }, "aPosition"),
            (SceGxmVertexAttribute{
                streamIndex: 0,
                offset: 8,
                format: SCE_GXM_ATTRIBUTE_FORMAT_U8 as u8,
                componentCount: 4,
                regIndex: 0,
            }, "aColor")
        ];

        let color_vertex_shader = shader_registry.register_vertex_shader(
            "color",
            COLOR_VERTEX_GXP,
            vertex_stride,
            &vertex_attributes
        ).unwrap();
        let texture_vertex_shader = shader_registry.register_vertex_shader(
            "texture",
            TEXTURE_VERTEX_GXP,
            vertex_stride,
            &vertex_attributes
        ).unwrap();
        let clear_vertex_shader = shader_registry.register_vertex_shader(
            "clear",
            CLEAR_VERTEX_GXP,
            vertex_stride,
            &vertex_attributes
        ).unwrap();
        
        let color_fragment_shader = shader_registry.register_fragment_shader("color", COLOR_FRAGMENT_GXP).unwrap();
        let bitmap_fragment_shader = shader_registry.register_fragment_shader("bitmap", BITMAP_FRAGMENT_GXP).unwrap();
        let gradient_fragment_shader = shader_registry.register_fragment_shader("gradient", GRADIENT_FRAGMENT_GXP).unwrap();
        let clear_fragment_shader = shader_registry.register_fragment_shader("clear", CLEAR_FRAGMENT_GXP).unwrap();

        let color_shader = shader_registry.register_shader(color_vertex_shader, color_fragment_shader).unwrap();
        let bitmap_shader = shader_registry.register_shader(texture_vertex_shader, bitmap_fragment_shader).unwrap();
        let gradient_shader = shader_registry.register_shader(texture_vertex_shader, gradient_fragment_shader).unwrap();
        let clear_shader = shader_registry.register_shader(clear_vertex_shader, clear_fragment_shader).unwrap();

        let quad_verticies = gxm::Buffer::from_slice(gxm::HeapType::LPDDR_R, &[
            Vertex {
                position: [0.0, 0.0],
                color: 0xffff_ffff,
            },
            Vertex {
                position: [1.0, 0.0],
                color: 0xffff_ffff,
            },
            Vertex {
                position: [1.0, 1.0],
                color: 0xffff_ffff,
            },
            Vertex {
                position: [0.0, 1.0],
                color: 0xffff_ffff,
            },
        ]);

        let quad_indicies = gxm::Buffer::from_slice(gxm::HeapType::LPDDR_R, &[0u32, 1, 2, 3]);

        let viewport = ViewportDimensions {
            width: display.width as u32,
            height: display.height as u32,
            scale_factor: 1.0,
        };

        let mut renderer = Self {
            context,
            display,
            shader_registry,

            color_shader,
            bitmap_shader,
            gradient_shader,
            clear_shader,

            msaa_sample_count,

            shape_tessellator: ShapeTessellator::new(),

            quad_verticies,
            quad_indicies,

            renderbuffer_width: 1,
            renderbuffer_height: 1,
            view_matrix: [[0.0; 4]; 4],

            mask_state: MaskState::NoMask,
            num_masks: 0,
            mask_state_dirty: true,
            is_transparent,
            blend_state: (0,0,0),

            blend_modes: vec![],

            viewport_scale_factor: 1.0,
            viewport,
            viewport_dirty: true,
        };

        renderer.push_blend_mode(RenderBlendMode::Builtin(BlendMode::Normal));

        Ok(renderer)
    }

    fn register_shape_internal(
        &mut self,
        shape: DistilledShape,
        bitmap_source: &dyn BitmapSource,
    ) -> Result<Vec<Draw>, Error> {
        use ruffle_render::tessellator::DrawType as TessDrawType;

        let lyon_mesh = self
            .shape_tessellator
            .tessellate_shape(shape, bitmap_source);

        let mut draws = Vec::with_capacity(lyon_mesh.draws.len());
        for draw in lyon_mesh.draws {
            let num_indices = draw.indices.len() as i32;
            let num_mask_indices = draw.mask_index_count as i32;

            // allocate verticies on the gpu memory
            let mut vertices: gxm::Buffer<Vertex> = gxm::Buffer::new(
                gxm::HeapType::LPDDR_R,
                draw.vertices.len(),
                4
            );
            let verticies_slice = vertices.as_mut_slice();
            for (i, vertex) in draw.vertices.into_iter().map(Vertex::from).enumerate() {
                verticies_slice[i] = vertex;
            }

            let indicies = gxm::Buffer::from_slice(gxm::HeapType::LPDDR_R, draw.indices.as_slice());

            draws.push(match draw.draw_type {
                TessDrawType::Color => Draw {
                    draw_type: DrawType::Color,
                    vertex_buffer: vertices,
                    index_buffer: indicies,
                    num_indices,
                    num_mask_indices,
                },
                TessDrawType::Gradient { matrix, gradient } => Draw {
                    draw_type: DrawType::Gradient(Box::new(Gradient::new(
                        lyon_mesh.gradients[gradient].clone(), // TODO: Gradient deduplication
                        matrix,
                    ))),
                    vertex_buffer: vertices,
                    index_buffer: indicies,
                    num_indices,
                    num_mask_indices,
                },
                TessDrawType::Bitmap(bitmap) => Draw {
                    draw_type: DrawType::Bitmap(BitmapDraw {
                        matrix: bitmap.matrix,
                        handle: bitmap_source.bitmap_handle(bitmap.bitmap_id, self),
                        is_smoothed: bitmap.is_smoothed,
                        is_repeating: bitmap.is_repeating,
                    }),
                    vertex_buffer: vertices,
                    index_buffer: indicies,
                    num_indices,
                    num_mask_indices,
                },
            });
        }

        Ok(draws)
    }

    pub fn width(&self) -> u32 {
        self.display.width
    }

    pub fn height(&self) -> u32 {
        self.display.height
    }

    fn set_stencil_state(&mut self) {
        // Set stencil state for masking, if necessary.
        if self.mask_state_dirty {
            match self.mask_state {
                MaskState::NoMask => {
                    self.context.set_front_stencil_func(
                        SCE_GXM_STENCIL_FUNC_ALWAYS,
                        SCE_GXM_STENCIL_OP_KEEP,
                        SCE_GXM_STENCIL_OP_KEEP,
                        SCE_GXM_STENCIL_OP_KEEP,
                        0xff,
                        0xff
                    );
                }
                MaskState::DrawMaskStencil => {
                    self.context.set_front_stencil_func(
                        SCE_GXM_STENCIL_FUNC_EQUAL,
                        SCE_GXM_STENCIL_OP_KEEP,
                        SCE_GXM_STENCIL_OP_KEEP,
                        SCE_GXM_STENCIL_OP_INCR,
                        (self.num_masks - 1) as u8,
                        0xff
                    );
                }
                MaskState::DrawMaskedContent => {
                    self.context.set_front_stencil_func(
                        SCE_GXM_STENCIL_FUNC_EQUAL,
                        SCE_GXM_STENCIL_OP_KEEP,
                        SCE_GXM_STENCIL_OP_KEEP,
                        SCE_GXM_STENCIL_OP_KEEP,
                        self.num_masks as u8,
                        0xff
                    );
                }
                MaskState::ClearMaskStencil => {
                    self.context.set_front_stencil_func(
                        SCE_GXM_STENCIL_FUNC_EQUAL,
                        SCE_GXM_STENCIL_OP_KEEP,
                        SCE_GXM_STENCIL_OP_KEEP,
                        SCE_GXM_STENCIL_OP_DECR,
                        self.num_masks as u8,
                        0xff
                    );
                }
            }
        }
    }

    fn apply_blend_mode(&mut self, mode: RenderBlendMode) {
        self.blend_state = match mode {
            RenderBlendMode::Builtin(BlendMode::Normal) => {
                // src + (1-a)
                (SCE_GXM_BLEND_FUNC_ADD as u8, SCE_GXM_BLEND_FACTOR_ONE as u8, SCE_GXM_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA as u8)
            }
            RenderBlendMode::Builtin(BlendMode::Add) => {
                // src + dst
                (SCE_GXM_BLEND_FUNC_ADD as u8, SCE_GXM_BLEND_FACTOR_ONE as u8, SCE_GXM_BLEND_FACTOR_ONE as u8)
            }
            RenderBlendMode::Builtin(BlendMode::Subtract) => {
                // dst - src
                (SCE_GXM_BLEND_FUNC_REVERSE_SUBTRACT as u8, SCE_GXM_BLEND_FACTOR_ONE as u8, SCE_GXM_BLEND_FACTOR_ONE as u8)
            }
            _ => {
                // TODO: Unsupported blend mode. Default to normal for now.
                (SCE_GXM_BLEND_FUNC_ADD as u8, SCE_GXM_BLEND_FACTOR_ONE as u8, SCE_GXM_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA as u8)
            }
        };
    }

    fn begin_frame(&mut self, clear: Color) {
        self.mask_state = MaskState::NoMask;
        self.num_masks = 0;
        self.mask_state_dirty = true;

        self.context.begin_scene(&mut self.display);
        if self.viewport_dirty {
            //self.viewport_dirty = false;
            let width = self.viewport.width as i32;
            let height = self.viewport.height as i32;
            self.context.set_viewport(0, 0, width, height);

            // Build view matrix based on canvas size.

            self.view_matrix = [
                [1.0 / width as f32, 0.0, 0.0, 0.0],   // Scale X, translate X to [-1, 1]
                [0.0, 1.0 / height as f32, 0.0, 1.0],   // Scale Y, flip Y, translate Y to [1, -1]
                [0.0, 0.0, 1.0, 0.0],                   // No depth scaling
                [0.0, 0.0, 0.0, 1.0],                   // Homogeneous coordinates
            ];
        }

        self.set_stencil_state();
        self.clear_screen(clear);
    }

    fn end_frame(&mut self) {
        self.context.end_scene();
    }

    pub fn swap(&mut self) {
        self.display.swap_buffers();
    }

    fn push_blend_mode(&mut self, blend: RenderBlendMode) {
        if !same_blend_mode(self.blend_modes.last(), &blend) {
            self.apply_blend_mode(blend.clone());
        }
        self.blend_modes.push(blend);
    }
    
    fn pop_blend_mode(&mut self) {
        let old = self.blend_modes.pop();
        // We never pop our base 'BlendMode::Normal'
        let current = self
            .blend_modes
            .last()
            .unwrap_or(&RenderBlendMode::Builtin(BlendMode::Normal));
        if !same_blend_mode(old.as_ref(), current) {
            self.apply_blend_mode(current.clone());
        }
    } 

    fn use_shader_program(&mut self, shader_type: ShaderProgramType) -> ShaderProgram {
        let shader_id = match shader_type {
            ShaderProgramType::Bitmap => self.bitmap_shader,
            ShaderProgramType::Color => self.color_shader,
            ShaderProgramType::Gradient => self.gradient_shader,
            ShaderProgramType::Clear => self.clear_shader,
        };

        let shader = self.shader_registry.get_shader(shader_id, self.blend_state).expect("missing shader {shader_type}");
        let uniform_buffers = self.context.use_program(&shader);

        ShaderProgram::new(shader, uniform_buffers).unwrap()
    }

    fn clear_screen(&mut self, clear: Color) {
        let program = self.use_shader_program(ShaderProgramType::Clear);

        program.uniform4fv(ShaderUniform::Color, &[clear.r as f32, clear.g as f32, clear.b as f32, 1.]);

        self.context.set_vertex_stream(&self.quad_verticies);

        self.context.draw(
            SCE_GXM_PRIMITIVE_TRIANGLE_FAN,
            SCE_GXM_INDEX_FORMAT_U32,
            &self.quad_indicies,
            0
        );
    }

    fn draw_quad<const MODE: u32, const COUNT: i32>(&mut self, color: Color, matrix: Matrix) {
        let world_matrix = [
            [matrix.a, matrix.b, 0.0, 0.0],
            [matrix.c, matrix.d, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [
                matrix.tx.to_pixels() as f32,
                matrix.ty.to_pixels() as f32,
                0.0,
                1.0,
            ],
        ];

        let mult_color = [
            color.r as f32,
            color.g as f32,
            color.b as f32,
            color.a as f32,
        ];
        let add_color = [0.0; 4];

        self.set_stencil_state();

        let program = self.use_shader_program(ShaderProgramType::Color);

        program.uniform_matrix4fv(ShaderUniform::ViewMatrix, &self.view_matrix);

        program.uniform_matrix4fv(ShaderUniform::WorldMatrix, &world_matrix);

        program.uniform4fv(ShaderUniform::MultColor, &mult_color);

        program.uniform4fv(ShaderUniform::AddColor, &add_color);

        self.context.set_vertex_stream(&self.quad_verticies);

        self.context.draw(
            MODE,
            SCE_GXM_INDEX_FORMAT_U32,
            &self.quad_indicies,
            COUNT.max(0) as usize
        );
    }
}

fn same_blend_mode(first: Option<&RenderBlendMode>, second: &RenderBlendMode) -> bool {
    match (first, second) {
        (Some(RenderBlendMode::Builtin(old)), RenderBlendMode::Builtin(new)) => old == new,
        _ => false,
    }
}

impl RenderBackend for GxmRenderBackend {
    fn render_offscreen(
        &mut self,
        _handle: BitmapHandle,
        _commands: CommandList,
        _quality: StageQuality,
        _bounds: PixelRegion,
    ) -> Option<Box<dyn SyncHandle>> {
        None
    }

    fn viewport_dimensions(&self) -> ViewportDimensions {
        ViewportDimensions {
            width: self.renderbuffer_width as u32,
            height: self.renderbuffer_height as u32,
            scale_factor: self.viewport_scale_factor,
        }
    }

    fn set_viewport_dimensions(&mut self, dimensions: ViewportDimensions) {
        println!("set_viewport_dimensions {:#?}", dimensions);

        // Setup GL viewport and renderbuffers clamped to reasonable sizes.
        // We don't use `.clamp()` here because `self.gl.drawing_buffer_width()` and
        // `self.gl.drawing_buffer_height()` return zero when the WebGL context is lost,
        // then an assertion error would be triggered.
        self.renderbuffer_width =
            (dimensions.width.max(1) as i32).min(self.display.width as i32);
        self.renderbuffer_height =
            (dimensions.height.max(1) as i32).min(self.display.height as i32);

        // Recreate framebuffers with the new size.
        self.viewport = dimensions;
        self.viewport_dirty = true;
        self.viewport_scale_factor = dimensions.scale_factor
    }

    fn register_shape(
        &mut self,
        shape: DistilledShape,
        bitmap_source: &dyn BitmapSource,
    ) -> ShapeHandle {
        let mesh = match self.register_shape_internal(shape, bitmap_source) {
            Ok(draws) => Mesh {
                draws,
            },
            Err(e) => {
                log::error!("Couldn't register shape: {:?}", e);
                Mesh {
                    draws: vec![],
                }
            }
        };
        ShapeHandle(Arc::new(mesh))
    }

    fn submit_frame(
        &mut self,
        clear: Color,
        commands: CommandList,
        cache_entries: Vec<BitmapCacheEntry>,
    ) {
        if !cache_entries.is_empty() {
            panic!("Bitmap caching is unavailable on the webgl backend");
        }
        self.begin_frame(clear);
        commands.execute(self);
        self.end_frame();
    }

    fn register_bitmap(&mut self, bitmap: Bitmap) -> Result<BitmapHandle, BitmapError> {
        let (tex_format, bitmap) = match bitmap.format() {
            BitmapFormat::Rgb => (SCE_GXM_TEXTURE_FORMAT_U8U8U8_RGB, bitmap),
            BitmapFormat::Rgba => (SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_RGBA, bitmap),
            BitmapFormat::Yuv420p => (SCE_GXM_TEXTURE_FORMAT_YUV420P3_CSC0, bitmap),
            BitmapFormat::Yuva420p => (SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_RGBA, bitmap.to_rgba())
        };

        let mut texture = gxm::Texture::new(
            bitmap.width(),
            bitmap.height(),
            tex_format,
            bitmap.data()
        );

        texture.set_filter(SCE_GXM_TEXTURE_FILTER_LINEAR, SCE_GXM_TEXTURE_FILTER_LINEAR);
        texture.set_addr_mode(SCE_GXM_TEXTURE_ADDR_CLAMP);

        Ok(BitmapHandle(Arc::new(RegistryData {
            width: bitmap.width(),
            height: bitmap.height(),
            texture,
        })))
    }

    fn update_texture(
        &mut self,
        handle: &BitmapHandle,
        bitmap: Bitmap,
        _region: PixelRegion,
    ) -> Result<(), BitmapError> {
        let reg_data: &mut RegistryData = as_registry_data(handle);

        let (_tex_format, bitmap) = match bitmap.format() {
            BitmapFormat::Rgb => (SCE_GXM_TEXTURE_FORMAT_U8U8U8_RGB, bitmap),
            BitmapFormat::Rgba => (SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_RGBA, bitmap),
            BitmapFormat::Yuv420p => (SCE_GXM_TEXTURE_FORMAT_YUV420P3_CSC0, bitmap),
            BitmapFormat::Yuva420p => (SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_RGBA, bitmap.to_rgba())
        };

        reg_data.texture.texture_data.copy_from_slice(bitmap.data());
        Ok(())
    }

    fn create_context3d(
        &mut self,
        _profile: Context3DProfile,
    ) -> Result<Box<dyn Context3D>, BitmapError> {
        Err(BitmapError::Unimplemented("createContext3D".into()))
    }
    fn context3d_present(&mut self, _context: &mut dyn Context3D) -> Result<(), BitmapError> {
        Err(BitmapError::Unimplemented("Context3D.present".into()))
    }

    fn debug_info(&self) -> Cow<'static, str> {
        let mut result = vec![];
        result.push("Renderer: gxm".to_string());
        result.push(format!("Surface samples: {} x ", self.msaa_sample_count));
        result.push(format!(
            "Surface size: {} x {}",
            self.renderbuffer_width, self.renderbuffer_height
        ));

        Cow::Owned(result.join("\n"))
    }

    fn name(&self) -> &'static str {
        "gxm"
    }

    fn set_quality(&mut self, _quality: StageQuality) {}

    fn compile_pixelbender_shader(
        &mut self,
        _shader: ruffle_render::pixel_bender::PixelBenderShader,
    ) -> Result<ruffle_render::pixel_bender::PixelBenderShaderHandle, BitmapError> {
        Err(BitmapError::Unimplemented(
            "compile_pixelbender_shader".into(),
        ))
    }

    fn resolve_sync_handle(
        &mut self,
        _handle: Box<dyn SyncHandle>,
        _with_rgba: RgbaBufRead,
    ) -> Result<(), ruffle_render::error::Error> {
        Err(ruffle_render::error::Error::Unimplemented(
            "Sync handle resolution".into(),
        ))
    }

    fn run_pixelbender_shader(
        &mut self,
        _handle: ruffle_render::pixel_bender::PixelBenderShaderHandle,
        _arguments: &[ruffle_render::pixel_bender::PixelBenderShaderArgument],
        _target: &PixelBenderTarget,
    ) -> Result<PixelBenderOutput, BitmapError> {
        Err(BitmapError::Unimplemented("run_pixelbender_shader".into()))
    }

    fn create_empty_texture(
        &mut self,
        width: u32,
        height: u32,
    ) -> Result<BitmapHandle, BitmapError> {
        let mut texture_data: Vec<u8> = Vec::new();
        texture_data.resize((width*height*3) as usize, 0);

        let texture = gxm::Texture::new(
            width,
            height,
            SCE_GXM_TEXTURE_FORMAT_U8U8U8_RGB,
            texture_data.as_slice()
        );

        Ok(BitmapHandle(Arc::new(RegistryData {
            width,
            height,
            texture
        })))
    }
}

impl CommandHandler for GxmRenderBackend {
    fn render_bitmap(
        &mut self,
        bitmap: BitmapHandle,
        transform: Transform,
        smoothing: bool,
        pixel_snapping: PixelSnapping,
    ) {
        self.set_stencil_state();
        let entry = as_registry_data(&bitmap);

        // Scale the quad to the bitmap's dimensions.
        let mut matrix = transform.matrix;
        pixel_snapping.apply(&mut matrix);
        matrix *= Matrix::scale(entry.width as f32, entry.height as f32);

        let world_matrix = [
            [matrix.a, matrix.c, 0.0, matrix.tx.to_pixels() as f32], // First row: scale_x, rotate_skew_1, z=0, translate_x
            [matrix.b, matrix.d, 0.0, matrix.ty.to_pixels() as f32], // Second row: rotate_skew_0, scale_y, z=0, translate_y
            [0.0, 0.0, 1.0, 0.0],                                  // Third row: no transformation in z-axis
            [0.0, 0.0, 0.0, 1.0],                                  // Fourth row: homogeneous coordinates
        ];

        let mult_color = transform.color_transform.mult_rgba_normalized();
        let add_color = transform.color_transform.add_rgba_normalized();

        let program = self.use_shader_program(ShaderProgramType::Bitmap);

        program.uniform_matrix4fv(ShaderUniform::ViewMatrix, &self.view_matrix);

        program.uniform_matrix4fv(ShaderUniform::WorldMatrix, &world_matrix);

        program.uniform4fv(ShaderUniform::MultColor, &mult_color);

        program.uniform4fv(ShaderUniform::AddColor, &add_color);

        let bitmap_matrix: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        program.uniform_matrix3fv(ShaderUniform::TextureMatrix, &bitmap_matrix);

        let filter = if smoothing { SCE_GXM_TEXTURE_FILTER_LINEAR } else { SCE_GXM_TEXTURE_FILTER_POINT };
        entry.texture.set_filter(filter, filter);
        entry.texture.set_addr_mode(SCE_GXM_TEXTURE_ADDR_CLAMP);
        self.context.set_texture(&entry.texture);

        self.context.set_vertex_stream(&self.quad_verticies);

        self.context.draw(
            SCE_GXM_PRIMITIVE_TRIANGLE_FAN,
            SCE_GXM_INDEX_FORMAT_U32,
            &self.quad_indicies,
            0
        );
    }

    fn render_shape(&mut self, shape: ShapeHandle, transform: Transform) {
        let world_matrix = [
            [transform.matrix.a, transform.matrix.b, 0.0, 0.0],
            [transform.matrix.c, transform.matrix.d, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [
                transform.matrix.tx.to_pixels() as f32,
                transform.matrix.ty.to_pixels() as f32,
                0.0,
                1.0,
            ],
        ];

        let mult_color = transform.color_transform.mult_rgba_normalized();
        let add_color = transform.color_transform.add_rgba_normalized();

        self.set_stencil_state();

        let mesh = as_mesh(&shape);
        for draw in &mesh.draws {
            // Ignore strokes when drawing a mask stencil.
            let num_indices = if self.mask_state != MaskState::DrawMaskStencil
                && self.mask_state != MaskState::ClearMaskStencil
            {
                draw.num_indices
            } else {
                draw.num_mask_indices
            };
            if num_indices == 0 {
                continue;
            }

            let program = self.use_shader_program(ShaderProgramType::from_draw_type(&draw.draw_type));

            program.uniform_matrix4fv(ShaderUniform::ViewMatrix, &self.view_matrix);

            program.uniform_matrix4fv(ShaderUniform::WorldMatrix, &world_matrix);

            program.uniform4fv(ShaderUniform::MultColor, &mult_color);

            program.uniform4fv(ShaderUniform::AddColor, &add_color);

            // Set shader specific uniforms.
            match &draw.draw_type {
                DrawType::Color => (),
                DrawType::Gradient(gradient) => {
                    program.uniform_matrix3fv(
                        ShaderUniform::TextureMatrix,
                        &gradient.matrix,
                    );
                    program.uniform1f(ShaderUniform::FocalPoint, gradient.focal_point);
                    program.set_uniform_data(ShaderUniform::Gradient, &[
                        gradient.interpolation as u8, // x
                        gradient.gradient_type as u8, // y
                        gradient.repeat_mode as u8, // z
                    ]);
                    self.context.set_texture(&gradient.texture);
                    //program.dump_uniform_data();
                }
                DrawType::Bitmap(bitmap) => {
                    let texture = match &bitmap.handle {
                        Some(handle) => &mut as_registry_data(handle).texture,
                        None => {
                            log::warn!("Tried to render a handleless bitmap");
                            continue;
                        }
                    };

                    program.uniform_matrix3fv(
                        ShaderUniform::TextureMatrix,
                        &bitmap.matrix,
                    );

                    self.context.set_texture(texture);
                    let filter = if bitmap.is_smoothed { SCE_GXM_TEXTURE_FILTER_LINEAR } else { SCE_GXM_TEXTURE_FILTER_POINT };
                    texture.set_filter(filter, filter);
                    let wrap = if bitmap.is_repeating { SCE_GXM_TEXTURE_ADDR_REPEAT } else { SCE_GXM_TEXTURE_ADDR_CLAMP };
                    texture.set_addr_mode(wrap);
                }
            }

            self.context.set_vertex_stream(&draw.vertex_buffer);
            self.context.draw(
                SCE_GXM_PRIMITIVE_TRIANGLES,
                SCE_GXM_INDEX_FORMAT_U32,
                &draw.index_buffer,
                0
            );
        }
    }

    fn render_stage3d(&mut self, _bitmap: BitmapHandle, _transform: Transform) {
        panic!("Stage3D should not have been created on GXM backend")
    }

    fn draw_rect(&mut self, color: Color, matrix: Matrix) {
        self.draw_quad::<{ SCE_GXM_PRIMITIVE_TRIANGLE_FAN }, -1>(color, matrix)
    }

    fn draw_line(&mut self, color: Color, mut matrix: Matrix) {
        matrix.tx += Twips::HALF;
        matrix.ty += Twips::HALF;
        self.draw_quad::<{  SCE_GXM_PRIMITIVE_LINES /*gl::LINE_STRIP*/ }, 2>(color, matrix)
    }

    fn draw_line_rect(&mut self, color: Color, mut matrix: Matrix) {
        matrix.tx += Twips::HALF;
        matrix.ty += Twips::HALF;
        self.draw_quad::<{ SCE_GXM_PRIMITIVE_LINES /*gl::LINE_LOOP*/ }, -1>(color, matrix)
    }

    fn push_mask(&mut self) {
        debug_assert!(
            self.mask_state == MaskState::NoMask || self.mask_state == MaskState::DrawMaskedContent
        );
        self.num_masks += 1;
        self.mask_state = MaskState::DrawMaskStencil;
        self.mask_state_dirty = true;
    }

    fn activate_mask(&mut self) {
        debug_assert!(self.num_masks > 0 && self.mask_state == MaskState::DrawMaskStencil);
        self.mask_state = MaskState::DrawMaskedContent;
        self.mask_state_dirty = true;
    }

    fn deactivate_mask(&mut self) {
        debug_assert!(self.num_masks > 0 && self.mask_state == MaskState::DrawMaskedContent);
        self.mask_state = MaskState::ClearMaskStencil;
        self.mask_state_dirty = true;
    }

    fn pop_mask(&mut self) {
        debug_assert!(self.num_masks > 0 && self.mask_state == MaskState::ClearMaskStencil);
        self.num_masks -= 1;
        self.mask_state = if self.num_masks == 0 {
            MaskState::NoMask
        } else {
            MaskState::DrawMaskedContent
        };
        self.mask_state_dirty = true;
    }

    fn blend(&mut self, commands: CommandList, blend: RenderBlendMode) {
        self.push_blend_mode(blend);
        commands.execute(self);
        self.pop_blend_mode();
    }
}

#[derive(Debug)]
struct Gradient {
    matrix: [[f32; 3]; 3],
    gradient_type: i32,
    repeat_mode: i32,
    focal_point: f32,
    interpolation: swf::GradientInterpolation,
    texture: gxm::Texture,
}

const GRADIENT_SIZE: usize = 0x100;

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn srgb_to_linear(color: f32) -> f32 {
    if color <= 0.04045 {
        color / 12.92
    } else {
        f32::powf((color + 0.055) / 1.055, 2.4)
    }
}

impl Gradient {
    fn new(gradient: TessGradient, matrix: [[f32; 3]; 3]) -> Self {
        let colors = if gradient.records.is_empty() {
            [0; GRADIENT_SIZE * 4]
        } else {
            let mut colors = [0; GRADIENT_SIZE * 4];

            let convert = if gradient.interpolation == swf::GradientInterpolation::LinearRgb {
                |c| srgb_to_linear(c / 255.0) * 255.0
            } else {
                |c| c
            };

            for t in 0..GRADIENT_SIZE {
                let mut last = 0;
                let mut next = 0;

                for (i, record) in gradient.records.iter().enumerate().rev() {
                    if (record.ratio as usize) < t {
                        last = i;
                        next = (i + 1).min(gradient.records.len() - 1);
                        break;
                    }
                }
                assert!(last == next || last + 1 == next);

                let last_record = &gradient.records[last];
                let next_record = &gradient.records[next];

                let a = if next == last {
                    // this can happen if we are before the first gradient record, or after the last one
                    0.0
                } else {
                    (t as f32 - last_record.ratio as f32)
                        / (next_record.ratio as f32 - last_record.ratio as f32)
                };
                colors[t * 4] = lerp(
                    convert(last_record.color.r as f32),
                    convert(next_record.color.r as f32),
                    a,
                ) as u8;
                colors[(t * 4) + 1] = lerp(
                    convert(last_record.color.g as f32),
                    convert(next_record.color.g as f32),
                    a,
                ) as u8;
                colors[(t * 4) + 2] = lerp(
                    convert(last_record.color.b as f32),
                    convert(next_record.color.b as f32),
                    a,
                ) as u8;
                colors[(t * 4) + 3] = lerp(
                    last_record.color.a as f32,
                    next_record.color.a as f32,
                    a,
                ) as u8;
            }

            colors
        };

        let texture = gxm::Texture::new(
            GRADIENT_SIZE as u32,
            1,
            SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_RGBA,
            &colors
        );

        Self {
            matrix,
            gradient_type: gradient.gradient_type as i32,
            repeat_mode: gradient.repeat_mode as i32,
            focal_point: gradient.focal_point.to_f32().clamp(-0.98, 0.98),
            interpolation: gradient.interpolation,
            texture,
        }
    }
}

#[derive(Clone, Debug)]
struct BitmapDraw {
    matrix: [[f32; 3]; 3],
    handle: Option<BitmapHandle>,
    is_repeating: bool,
    is_smoothed: bool,
}

#[derive(Debug)]
struct Mesh {
    draws: Vec<Draw>,
}

impl ShapeHandleImpl for Mesh {}

fn as_mesh(handle: &ShapeHandle) -> &Mesh {
    <dyn ShapeHandleImpl>::downcast_ref(&*handle.0).expect("Shape handle must be a GXM.ofw ShapeData")
}


#[allow(dead_code)]
#[derive(Debug)]
struct Draw {
    draw_type: DrawType,
    vertex_buffer: gxm::Buffer<Vertex>,
    index_buffer: gxm::Buffer<u32>,
    num_indices: i32,
    num_mask_indices: i32,
}

#[derive(Debug)]
enum DrawType {
    Color,
    Gradient(Box<Gradient>),
    Bitmap(BitmapDraw),
}

enum ShaderProgramType {
    Color,
    Bitmap,
    Gradient,
    Clear
}

impl ShaderProgramType {
    fn from_draw_type(draw_type: &DrawType) -> ShaderProgramType {
        match draw_type {
            DrawType::Bitmap(..) => ShaderProgramType::Bitmap,
            DrawType::Color => ShaderProgramType::Color,
            DrawType::Gradient(_) => ShaderProgramType::Gradient,
        }
    }
}

// Because the shaders are currently simple and few in number, we are using a
// straightforward shader model. We maintain an enum of every possible uniform,
// and each shader tries to grab the location of each uniform.
struct ShaderProgram {
    program: Rc<gxm::ShaderProgram>,
    uniform_buffers: (*mut c_void, *mut c_void)
}

// These should match the uniform names in the shaders.
const NUM_UNIFORMS: usize = 8;
const UNIFORM_NAMES: [&str; NUM_UNIFORMS] = [
    "worldMatrix",
    "viewMatrix",
    "multColor",
    "addColor",
    "uMatrix",
    "focal_point",
    "gradient",
    "uColor",
];

enum ShaderUniform {
    WorldMatrix = 0,
    ViewMatrix,
    MultColor,
    AddColor,
    TextureMatrix,
    FocalPoint,
    Gradient,
    Color
}

impl ShaderProgram {
    fn new(
        program: Rc<gxm::ShaderProgram>,
        uniform_buffers: gxm::UniformBuffers
    ) -> Result<Self, Error> {
        Ok(ShaderProgram {
            program: program,
            uniform_buffers
        })
    }

    fn uniform1f(&self, uniform: ShaderUniform, value: f32) {
        self.program.as_ref().set_uniform(
            uniform as usize,
            &[value],
            self.uniform_buffers,
        );
    }

    fn uniform1fv(&self, uniform: ShaderUniform, values: &[f32]) {
        self.program.set_uniform(
            uniform as usize,
            values,
            self.uniform_buffers,
        );
    }

    fn uniform1i(&self, uniform: ShaderUniform, value: u32) {
        self.program.set_uniform(
            uniform as usize,
            &[f32::from_bits(value)],
            self.uniform_buffers,
        );
    }

    fn uniform4fv(&self, uniform: ShaderUniform, values: &[f32]) {
        self.program.set_uniform(
            uniform as usize,
            &values,
            self.uniform_buffers,
        );
    }

    fn uniform_matrix3fv(&self, uniform: ShaderUniform, values: &[[f32; 3]; 3]) {
        self.program.set_uniform(
            uniform as usize,
            &values.as_flattened(),
            self.uniform_buffers,
        );
    }

    fn uniform_matrix4fv(&self, uniform: ShaderUniform, values: &[[f32; 4]; 4]) {
        self.program.set_uniform(
            uniform as usize,
            &values.as_flattened(),
            self.uniform_buffers,
        );
    }

    fn set_uniform_data<T: Copy>(&self, uniform: ShaderUniform, value: &T) {
        self.program.set_uniform_data(uniform as usize, value, self.uniform_buffers);
    }

    fn dump_uniform_data(&self) {
        self.program.dump_uniform_data(self.uniform_buffers);
    }
}
