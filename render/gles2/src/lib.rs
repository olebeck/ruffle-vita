#![deny(clippy::unwrap_used)]
// Remove this when we start using `Rc` when compiling for wasm
#![allow(clippy::arc_with_non_send_sync)]

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
use ruffle_render::shape_utils::{DistilledShape, GradientType};
use ruffle_render::tessellator::{
    Gradient as TessGradient, ShapeTessellator, Vertex as TessVertex,
};
use ruffle_render::transform::Transform;
use std::borrow::Cow;
use std::sync::Arc;
use swf::{BlendMode, Color, Twips};
use thiserror::Error;

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

const COLOR_VERTEX_GLSL: &str = include_str!("../shaders/color.vert");
const COLOR_FRAGMENT_GLSL: &str = include_str!("../shaders/color.frag");
const TEXTURE_VERTEX_GLSL: &str = include_str!("../shaders/texture.vert");
const GRADIENT_FRAGMENT_GLSL: &str = include_str!("../shaders/gradient.frag");
const BITMAP_FRAGMENT_GLSL: &str = include_str!("../shaders/bitmap.frag");
const NUM_VERTEX_ATTRIBUTES: u32 = 2;

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

pub struct GLES2RenderBackend {
    // The frame buffers used for resolving MSAA.
    msaa_buffers: Option<MsaaBuffers>,
    msaa_sample_count: u32,

    color_program: ShaderProgram,
    bitmap_program: ShaderProgram,
    gradient_program: ShaderProgram,

    shape_tessellator: ShapeTessellator,

    color_quad_draws: Vec<Draw>,
    bitmap_quad_draws: Vec<Draw>,

    mask_state: MaskState,
    num_masks: u32,
    mask_state_dirty: bool,
    is_transparent: bool,

    active_program: *const ShaderProgram,
    blend_modes: Vec<RenderBlendMode>,
    mult_color: Option<[f32; 4]>,
    add_color: Option<[f32; 4]>,

    renderbuffer_width: i32,
    renderbuffer_height: i32,
    view_matrix: [[f32; 4]; 4],

    drawing_buffer_width: i32,
    drawing_buffer_height: i32,

    // This is currently unused - we just hold on to it
    // to expose via `get_viewport_dimensions`
    viewport_scale_factor: f64,
}

#[derive(Debug)]
struct RegistryData {
    width: u32,
    height: u32,
    texture: gl::types::GLuint,
}

impl Drop for RegistryData {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteTextures(1, &self.texture as *const gl::types::GLuint);
        }
    }
}

impl BitmapHandleImpl for RegistryData {}

fn as_registry_data(handle: &BitmapHandle) -> &RegistryData {
    <dyn BitmapHandleImpl>::downcast_ref(&*handle.0)
        .expect("Bitmap handle must be webgl RegistryData")
}

const MAX_GRADIENT_COLORS: usize = 15;

impl GLES2RenderBackend {
    pub fn new(
        is_transparent: bool,
        quality: StageQuality,
        drawing_buffer_width: i32,
        drawing_buffer_height: i32,
    ) -> Result<Self, Error> {
        // Determine MSAA sample count.
        let mut msaa_sample_count = quality.sample_count().min(4);

        // Ensure that we don't exceed the max MSAA of this device.
        let mut max_samples: gl::types::GLint = 0;
        unsafe {
            gl::GetIntegerv(gl::MAX_SAMPLES, &mut max_samples);
        }
        if max_samples > 0 && (max_samples as u32) < msaa_sample_count {
            log::info!("Device only supports {}xMSAA", max_samples);
            msaa_sample_count = max_samples as u32;
        }

        let color_vertex = Self::compile_shader(gl::VERTEX_SHADER, COLOR_VERTEX_GLSL)?;
        let texture_vertex = Self::compile_shader(gl::VERTEX_SHADER, TEXTURE_VERTEX_GLSL)?;
        let color_fragment = Self::compile_shader(gl::FRAGMENT_SHADER, COLOR_FRAGMENT_GLSL)?;
        let bitmap_fragment = Self::compile_shader(gl::FRAGMENT_SHADER, BITMAP_FRAGMENT_GLSL)?;
        let gradient_fragment = Self::compile_shader(gl::FRAGMENT_SHADER, GRADIENT_FRAGMENT_GLSL)?;

        let color_program = ShaderProgram::new("color", color_vertex, color_fragment)?;
        let bitmap_program = ShaderProgram::new("bitmap", texture_vertex, bitmap_fragment)?;
        let gradient_program = ShaderProgram::new("gradient", texture_vertex, gradient_fragment)?;

        unsafe {
            gl::Enable(gl::BLEND);

            // Necessary to load RGB textures (alignment defaults to 4).
            gl::PixelStorei(gl::UNPACK_ALIGNMENT, 1);
        }

        let mut renderer = Self {
            msaa_buffers: None,
            msaa_sample_count,

            color_program,
            gradient_program,
            bitmap_program,

            shape_tessellator: ShapeTessellator::new(),

            color_quad_draws: vec![],
            bitmap_quad_draws: vec![],
            renderbuffer_width: 1,
            renderbuffer_height: 1,
            drawing_buffer_width: drawing_buffer_width,
            drawing_buffer_height: drawing_buffer_height,
            view_matrix: [[0.0; 4]; 4],

            mask_state: MaskState::NoMask,
            num_masks: 0,
            mask_state_dirty: true,
            is_transparent,

            active_program: std::ptr::null(),
            blend_modes: vec![],
            mult_color: None,
            add_color: None,

            viewport_scale_factor: 1.0,
        };

        renderer.push_blend_mode(RenderBlendMode::Builtin(BlendMode::Normal));

        let mut color_quad_mesh = renderer.build_quad_mesh(&renderer.color_program)?;
        let mut bitmap_quad_mesh = renderer.build_quad_mesh(&renderer.bitmap_program)?;
        renderer.color_quad_draws.append(&mut color_quad_mesh);
        renderer.bitmap_quad_draws.append(&mut bitmap_quad_mesh);

        renderer.set_viewport_dimensions(ViewportDimensions {
            width: drawing_buffer_width as u32,
            height: drawing_buffer_height as u32,
            scale_factor: 1.0,
        });

        Ok(renderer)
    }

    fn build_quad_mesh(&self, program: &ShaderProgram) -> Result<Vec<Draw>, Error> {
        let vao = self.create_vertex_array()?;

        unsafe {
            let mut vertex_buffer: u32 = 0;
            gl::GenBuffers(1, &mut vertex_buffer);
            gl::BindBuffer(gl::ARRAY_BUFFER, vertex_buffer);

            let verticies: &[u8] = bytemuck::cast_slice(&[
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
            gl::BufferData(
                gl::ARRAY_BUFFER,
                verticies.len() as isize,
                verticies.as_ptr() as *const gl::types::GLvoid,
                gl::STATIC_DRAW
            );

            let mut index_buffer: u32 = 0;
            gl::GenBuffers(1, &mut index_buffer);
            gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, index_buffer);

            let indicies: &[u8] = bytemuck::cast_slice(&[0u32, 1, 2, 3]);
            gl::BufferData(
                gl::ELEMENT_ARRAY_BUFFER,
                indicies.len() as isize,
                indicies.as_ptr() as *const gl::types::GLvoid,
                gl::STATIC_DRAW
            );

            if program.vertex_position_location != 0xffff_ffff {
                gl::VertexAttribPointer(
                    program.vertex_position_location,
                    2,
                    gl::FLOAT,
                    false as u8,
                    12,
                    0 as *const gl::types::GLvoid,
                );
                gl::EnableVertexAttribArray(program.vertex_position_location);
            }

            if program.vertex_color_location != 0xffff_ffff {
                gl::VertexAttribPointer(
                    program.vertex_color_location,
                    4,
                    gl::UNSIGNED_BYTE,
                    true as u8,
                    12,
                    8 as *const gl::types::GLvoid,
                );
                gl::EnableVertexAttribArray(program.vertex_color_location);
            }
            self.bind_vertex_array(None);
            for i in program.num_vertex_attributes..NUM_VERTEX_ATTRIBUTES {
                gl::DisableVertexAttribArray(i);
            }

            let mut draws = vec![];
            draws.push(Draw {
                draw_type: if program.program == self.bitmap_program.program {
                    DrawType::Bitmap(BitmapDraw {
                        matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        handle: None,
                        is_smoothed: true,
                        is_repeating: false,
                    })
                } else {
                    DrawType::Color
                },
                vao,
                vertex_buffer: Buffer {
                    buffer: vertex_buffer,
                },
                index_buffer: Buffer {
                    buffer: index_buffer,
                },
                num_indices: 4,
                num_mask_indices: 4,
            });
            Ok(draws)
        }
    }

    fn compile_shader(shader_type: u32, glsl_src: &str) -> Result<gl::types::GLuint, Error> {
        println!("compile_shader");
        unsafe {
            let shader = gl::CreateShader(shader_type);
            let shader_len = glsl_src.len() as i32;
            gl::ShaderSource(
                shader,
                1,
                &to_cstring!(glsl_src),
                &shader_len,
            );
            gl::CompileShader(shader);
            let log = get_shader_info_log(shader);
            if !log.is_empty() {
                println!("{}", log);
            }
            Ok(shader)
        }
    }

    fn build_msaa_buffers(&mut self) -> Result<(), Error> {
        unsafe {
            if self.msaa_sample_count <= 1 || true {
                gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
                gl::BindRenderbuffer(gl::RENDERBUFFER, 0);
                return Ok(());
            }

            // Delete previous buffers, if they exist.
            if let Some(msaa_buffers) = self.msaa_buffers.take() {
                gl::DeleteRenderbuffers(1, &msaa_buffers.color_renderbuffer);
                gl::DeleteRenderbuffers(1, &msaa_buffers.stencil_renderbuffer);
                gl::DeleteFramebuffers(1, &msaa_buffers.render_framebuffer);
                gl::DeleteFramebuffers(1, &msaa_buffers.color_framebuffer);
                gl::DeleteTextures(1, &msaa_buffers.framebuffer_texture);
            }

            // Create frame and render buffers.
            let mut render_framebuffer: u32 = 0;
            gl::GenFramebuffers(1, &mut render_framebuffer);
            let mut color_framebuffer: u32 = 0;
            gl::GenFramebuffers(1, &mut color_framebuffer);

            // Note for future self:
            // Whenever we support playing transparent movies,
            // switch this to RGBA and probably need to change shaders to all
            // be premultiplied alpha.
            let mut color_renderbuffer: u32 = 0;
            gl::GenRenderbuffers(1, &mut color_renderbuffer);
            gl::BindRenderbuffer(gl::RENDERBUFFER, color_renderbuffer);
            gl::RenderbufferStorageMultisample(
                gl::RENDERBUFFER,
                self.msaa_sample_count as i32,
                gl::RGBA8,
                self.renderbuffer_width,
                self.renderbuffer_height,
            );
            gl_check_error("renderbuffer_storage_multisample (color)")?;

            let mut stencil_renderbuffer: u32 = 0;
            gl::GenRenderbuffers(1, &mut stencil_renderbuffer);
            gl::BindRenderbuffer(gl::RENDERBUFFER, stencil_renderbuffer);
            gl::RenderbufferStorageMultisample(
                gl::RENDERBUFFER,
                self.msaa_sample_count as i32,
                gl::STENCIL_INDEX8,
                self.renderbuffer_width,
                self.renderbuffer_height,
            );
            gl_check_error("renderbuffer_storage_multisample (stencil)")?;

            gl::BindFramebuffer(gl::FRAMEBUFFER, render_framebuffer);
            gl::FramebufferRenderbuffer(
                gl::FRAMEBUFFER,
                gl::COLOR_ATTACHMENT0,
                gl::RENDERBUFFER,
                color_renderbuffer,
            );
            gl::FramebufferRenderbuffer(
                gl::FRAMEBUFFER,
                gl::STENCIL_ATTACHMENT,
                gl::RENDERBUFFER,
                stencil_renderbuffer,
            );

            let mut framebuffer_texture: u32 = 0;
            gl::GenTextures(1, &mut framebuffer_texture);
            gl::BindTexture(gl::TEXTURE_2D, framebuffer_texture);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as i32);
            gl::TexParameteri(
                gl::TEXTURE_2D,
                gl::TEXTURE_WRAP_S,
                gl::CLAMP_TO_EDGE as i32,
            );
            gl::TexParameteri(
                gl::TEXTURE_2D,
                gl::TEXTURE_WRAP_T,
                gl::CLAMP_TO_EDGE as i32,
            );
            gl::TexImage2D(
                gl::TEXTURE_2D,
                0,
                gl::RGBA as i32,
                self.renderbuffer_width,
                self.renderbuffer_height,
                0,
                gl::RGBA,
                gl::UNSIGNED_BYTE,
                0 as *const gl::types::GLvoid,
            );
            gl::BindTexture(gl::TEXTURE_2D, 0);

            gl::BindFramebuffer(gl::FRAMEBUFFER, color_framebuffer);
            gl::FramebufferTexture2D(
                gl::FRAMEBUFFER,
                gl::COLOR_ATTACHMENT0,
                gl::TEXTURE_2D,
                framebuffer_texture,
                0,
            );
            gl::BindFramebuffer(gl::FRAMEBUFFER, 0);

            self.msaa_buffers = Some(MsaaBuffers {
                color_renderbuffer,
                stencil_renderbuffer,
                render_framebuffer,
                color_framebuffer,
                framebuffer_texture,
            });

            Ok(())
        }
    }

    fn register_shape_internal(
        &mut self,
        shape: DistilledShape,
        bitmap_source: &dyn BitmapSource,
    ) -> Result<Vec<Draw>, Error> {
        unsafe {
            use ruffle_render::tessellator::DrawType as TessDrawType;

            let lyon_mesh = self
                .shape_tessellator
                .tessellate_shape(shape, bitmap_source);

            let mut draws = Vec::with_capacity(lyon_mesh.draws.len());
            for draw in lyon_mesh.draws {
                let num_indices = draw.indices.len() as i32;
                let num_mask_indices = draw.mask_index_count as i32;

                let vao = self.create_vertex_array()?;

                let mut vertex_buffer: u32 = 0;
                gl::GenBuffers(1,&mut vertex_buffer);
                gl::BindBuffer(gl::ARRAY_BUFFER, vertex_buffer);

                let vertices: Vec<_> = draw.vertices.into_iter().map(Vertex::from).collect();
                gl::BufferData(
                    gl::ARRAY_BUFFER,
                    (vertices.len() * std::mem::size_of::<Vertex>()) as isize,
                    vertices.as_ptr() as *const gl::types::GLvoid,
                    gl::STATIC_DRAW,
                );

                let mut index_buffer: u32 = 0;
                gl::GenBuffers(1,&mut index_buffer);
                gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, index_buffer);

                gl::BufferData(
                    gl::ELEMENT_ARRAY_BUFFER,
                    (draw.indices.len() * 4) as isize,
                    draw.indices.as_ptr() as *const gl::types::GLvoid,
                    gl::STATIC_DRAW,
                );

                let program = match draw.draw_type {
                    TessDrawType::Color => &self.color_program,
                    TessDrawType::Gradient { .. } => &self.gradient_program,
                    TessDrawType::Bitmap(_) => &self.bitmap_program,
                };

                // Unfortunately it doesn't seem to be possible to ensure that vertex attributes will be in
                // a guaranteed position between shaders in WebGL1 (no layout qualifiers in GLSL in OpenGL ES 1.0).
                // Attributes can change between shaders, even if the vertex layout is otherwise "the same".
                // This varies between platforms based on what the GLSL compiler decides to do.
                if program.vertex_position_location != 0xffff_ffff {
                    gl::VertexAttribPointer(
                        program.vertex_position_location,
                        2,
                        gl::FLOAT,
                        false as u8,
                        12,
                        0 as *const gl::types::GLvoid,
                    );
                    gl::EnableVertexAttribArray(program.vertex_position_location);
                }

                if program.vertex_color_location != 0xffff_ffff {
                    gl::VertexAttribPointer(
                        program.vertex_color_location,
                        4,
                        gl::UNSIGNED_BYTE,
                        true as u8,
                        12,
                        8 as *const gl::types::GLvoid,
                    );
                    gl::EnableVertexAttribArray(program.vertex_color_location);
                }

                let num_vertex_attributes = program.num_vertex_attributes;

                draws.push(match draw.draw_type {
                    TessDrawType::Color => Draw {
                        draw_type: DrawType::Color,
                        vao,
                        vertex_buffer: Buffer {
                            buffer: vertex_buffer,
                        },
                        index_buffer: Buffer {
                            buffer: index_buffer,
                        },
                        num_indices,
                        num_mask_indices,
                    },
                    TessDrawType::Gradient { matrix, gradient } => Draw {
                        draw_type: DrawType::Gradient(Box::new(Gradient::new(
                            lyon_mesh.gradients[gradient].clone(), // TODO: Gradient deduplication
                            matrix,
                        ))),
                        vao,
                        vertex_buffer: Buffer {
                            buffer: vertex_buffer,
                        },
                        index_buffer: Buffer {
                            buffer: index_buffer,
                        },
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
                        vao,
                        vertex_buffer: Buffer {
                            buffer: vertex_buffer,
                        },
                        index_buffer: Buffer {
                            buffer: index_buffer,
                        },
                        num_indices,
                        num_mask_indices,
                    },
                });

                self.bind_vertex_array(None);

                // Don't use 'program' here in order to satisfy the borrow checker
                for i in num_vertex_attributes..NUM_VERTEX_ATTRIBUTES {
                    gl::DisableVertexAttribArray(i);
                }
            }

            Ok(draws)
        }
    }

    /// Creates and binds a new VAO.
    fn create_vertex_array(&self) -> Result<u32, Error> {
        unsafe {
            let mut vao: u32 = 0;
            gl::GenVertexArrays(1, &mut vao);
            gl::BindVertexArray(vao);
            Ok(vao)
        }
    }

    /// Binds a VAO.
    fn bind_vertex_array(&self, vao: Option<gl::types::GLuint>) {
        unsafe {
            gl::BindVertexArray(vao.unwrap_or_default());
        }
    }

    fn set_stencil_state(&mut self) {
        unsafe {
            // Set stencil state for masking, if necessary.
            if self.mask_state_dirty {
                match self.mask_state {
                    MaskState::NoMask => {
                        gl::Disable(gl::STENCIL_TEST);
                        gl::ColorMask(1, 1, 1, 1);
                    }
                    MaskState::DrawMaskStencil => {
                        gl::Enable(gl::STENCIL_TEST);
                        gl::StencilFunc(gl::EQUAL, (self.num_masks - 1) as i32, 0xff);
                        gl::StencilOp(gl::KEEP, gl::KEEP, gl::INCR);
                        gl::ColorMask(0, 0, 0, 0);
                    }
                    MaskState::DrawMaskedContent => {
                        gl::Enable(gl::STENCIL_TEST);
                        gl::StencilFunc(gl::EQUAL, self.num_masks as i32, 0xff);
                        gl::StencilOp(gl::KEEP, gl::KEEP, gl::KEEP);
                        gl::ColorMask(1, 1, 1, 1);
                    }
                    MaskState::ClearMaskStencil => {
                        gl::Enable(gl::STENCIL_TEST);
                        gl::StencilFunc(gl::EQUAL, self.num_masks as i32, 0xff);
                        gl::StencilOp(gl::KEEP, gl::KEEP, gl::DECR);
                        gl::ColorMask(0, 0, 0, 0);
                    }
                }
            }
        }
    }

    fn apply_blend_mode(&mut self, mode: RenderBlendMode) {
        let (blend_op, src_rgb, dst_rgb) = match mode {
            RenderBlendMode::Builtin(BlendMode::Normal) => {
                // src + (1-a)
                (gl::FUNC_ADD, gl::ONE, gl::ONE_MINUS_SRC_ALPHA)
            }
            RenderBlendMode::Builtin(BlendMode::Add) => {
                // src + dst
                (gl::FUNC_ADD, gl::ONE, gl::ONE)
            }
            RenderBlendMode::Builtin(BlendMode::Subtract) => {
                // dst - src
                (gl::FUNC_REVERSE_SUBTRACT, gl::ONE, gl::ONE)
            }
            _ => {
                // TODO: Unsupported blend mode. Default to normal for now.
                (gl::FUNC_ADD, gl::ONE, gl::ONE_MINUS_SRC_ALPHA)
            }
        };
        unsafe {
            gl::BlendEquationSeparate(blend_op, gl::FUNC_ADD);
            gl::BlendFuncSeparate(src_rgb, dst_rgb, gl::ONE, gl::ONE_MINUS_SRC_ALPHA);
        }
    }

    fn begin_frame(&mut self, clear: Color) {
        self.active_program = std::ptr::null();
        self.mask_state = MaskState::NoMask;
        self.num_masks = 0;
        self.mask_state_dirty = true;

        self.mult_color = None;
        self.add_color = None;

        unsafe {
            // Bind to MSAA render buffer if using MSAA.
            if let Some(msaa_buffers) = &self.msaa_buffers {
                gl::BindFramebuffer(gl::FRAMEBUFFER, msaa_buffers.render_framebuffer);
            }

            gl::Viewport(0, 0, self.renderbuffer_width, self.renderbuffer_height);

            self.set_stencil_state();
            if self.is_transparent {
                gl::ClearColor(0.0, 0.0, 0.0, 0.0);
            } else {
                gl::ClearColor(
                    clear.r as f32 / 255.0,
                    clear.g as f32 / 255.0,
                    clear.b as f32 / 255.0,
                    clear.a as f32 / 255.0,
                );
            }
            gl::StencilMask(0xff);
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::STENCIL_BUFFER_BIT);
        }
    }

    fn end_frame(&mut self) {
        unsafe {
            // Resolve MSAA, if we're using it (WebGL2).
            if let Some(ref msaa_buffers) = &self.msaa_buffers {
                // Disable any remaining masking state.
                gl::Disable(gl::STENCIL_TEST);
                gl::ColorMask(1, 1, 1, 1);

                // Resolve the MSAA in the render buffer.
                gl::BindFramebuffer(
                    gl::READ_FRAMEBUFFER,
                    msaa_buffers.render_framebuffer,
                );
                gl::BindFramebuffer(gl::DRAW_FRAMEBUFFER, msaa_buffers.color_framebuffer);
                gl::BlitFramebuffer(
                    0,
                    0,
                    self.renderbuffer_width,
                    self.renderbuffer_height,
                    0,
                    0,
                    self.renderbuffer_width,
                    self.renderbuffer_height,
                    gl::COLOR_BUFFER_BIT,
                    gl::NEAREST,
                );

                // Render the resolved framebuffer texture to a quad on the screen.
                gl::BindFramebuffer(gl::FRAMEBUFFER, 0);

                gl::Viewport(
                    0,
                    0,
                    self.drawing_buffer_width,
                    self.drawing_buffer_height,
                );

                let program = &self.bitmap_program;
                gl::UseProgram(program.program);

                // Scale to fill screen.
                program.uniform_matrix4fv(
                    ShaderUniform::WorldMatrix,
                    &[
                        [2.0, 0.0, 0.0, 0.0],
                        [0.0, 2.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [-1.0, -1.0, 0.0, 1.0],
                    ],
                );
                program.uniform_matrix4fv(
                    ShaderUniform::ViewMatrix,
                    &[
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                );
                program.uniform4fv(ShaderUniform::MultColor, &[1.0, 1.0, 1.0, 1.0]);
                program.uniform4fv(ShaderUniform::AddColor, &[0.0, 0.0, 0.0, 0.0]);

                program.uniform_matrix3fv(
                    ShaderUniform::TextureMatrix,
                    &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                );

                // Bind the framebuffer texture.
                gl::ActiveTexture(gl::TEXTURE0);
                gl::BindTexture(gl::TEXTURE_2D, msaa_buffers.framebuffer_texture);
                program.uniform1i(ShaderUniform::BitmapTexture, 0);

                // Render the quad.
                let quad = &self.bitmap_quad_draws;
                self.bind_vertex_array(Some(quad[0].vao));
                gl::DrawElements(
                    gl::TRIANGLE_FAN,
                    quad[0].num_indices,
                    gl::UNSIGNED_INT,
                    0 as *const gl::types::GLvoid,
                );
            }
        }
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
            color.r as f32 * 255.0,
            color.g as f32 * 255.0,
            color.b as f32 * 255.0,
            color.a as f32 * 255.0,
        ];
        let add_color = [0.0; 4];

        self.set_stencil_state();

        let program = &self.color_program;

        // Set common render state, while minimizing unnecessary state changes.
        // TODO: Using designated layout specifiers in WebGL2/OpenGL ES 3, we could guarantee that uniforms
        // are in the same location between shaders, and avoid changing them unless necessary.
        if program as *const ShaderProgram != self.active_program {
            unsafe {
                gl::UseProgram(program.program);
            }
            self.active_program = program as *const ShaderProgram;

            program.uniform_matrix4fv(ShaderUniform::ViewMatrix, &self.view_matrix);

            self.mult_color = None;
            self.add_color = None;
        };

        self.color_program
            .uniform_matrix4fv(ShaderUniform::WorldMatrix, &world_matrix);
        if Some(mult_color) != self.mult_color {
            self.color_program
                .uniform4fv(ShaderUniform::MultColor, &mult_color);
            self.mult_color = Some(mult_color);
        }
        if Some(add_color) != self.add_color {
            self.color_program
                .uniform4fv(ShaderUniform::AddColor, &add_color);
            self.add_color = Some(add_color);
        }

        let quad = &self.color_quad_draws;
        self.bind_vertex_array(Some(quad[0].vao));

        let count = if COUNT < 0 {
            quad[0].num_indices
        } else {
            COUNT
        };
        unsafe {
            gl::DrawElements(MODE, count, gl::UNSIGNED_INT, 0 as *const gl::types::GLvoid);
        }
    }
}

fn same_blend_mode(first: Option<&RenderBlendMode>, second: &RenderBlendMode) -> bool {
    match (first, second) {
        (Some(RenderBlendMode::Builtin(old)), RenderBlendMode::Builtin(new)) => old == new,
        _ => false,
    }
}

impl RenderBackend for GLES2RenderBackend {
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
        // Build view matrix based on canvas size.
        self.view_matrix = [
            [1.0 / (dimensions.width as f32 / 2.0), 0.0, 0.0, 0.0],
            [0.0, -1.0 / (dimensions.height as f32 / 2.0), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0, 1.0],
        ];

        // Setup GL viewport and renderbuffers clamped to reasonable sizes.
        // We don't use `.clamp()` here because `self.gl.drawing_buffer_width()` and
        // `self.gl.drawing_buffer_height()` return zero when the WebGL context is lost,
        // then an assertion error would be triggered.
        self.renderbuffer_width =
            (dimensions.width.max(1) as i32).min(self.drawing_buffer_width);
        self.renderbuffer_height =
            (dimensions.height.max(1) as i32).min(self.drawing_buffer_height);

        // Recreate framebuffers with the new size.
        let _ = self.build_msaa_buffers();
        unsafe {
            gl::Viewport(0, 0, self.renderbuffer_width, self.renderbuffer_height);
        }
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
        let (format, bitmap) = match bitmap.format() {
            BitmapFormat::Rgb | BitmapFormat::Yuv420p => (gl::RGB, bitmap.to_rgb()),
            BitmapFormat::Rgba | BitmapFormat::Yuva420p => (gl::RGBA, bitmap.to_rgba()),
        };

        unsafe {
            let mut texture: u32 = 0;
            gl::GenTextures(1, &mut texture);
            gl::BindTexture(gl::TEXTURE_2D, texture);
            gl::TexImage2D(
                gl::TEXTURE_2D,
                0,
                format as i32,
                bitmap.width() as i32,
                bitmap.height() as i32,
                0,
                format,
                gl::UNSIGNED_BYTE,
                bitmap.data().as_ptr() as *const gl::types::GLvoid,
            );

            // You must set the texture parameters for non-power-of-2 textures to function in WebGL1.
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);

            Ok(BitmapHandle(Arc::new(RegistryData {
                width: bitmap.width(),
                height: bitmap.height(),
                texture,
            })))
        }
    }

    fn update_texture(
        &mut self,
        handle: &BitmapHandle,
        bitmap: Bitmap,
        _region: PixelRegion,
    ) -> Result<(), BitmapError> {
        let texture = as_registry_data(handle).texture;

        let (format, bitmap) = match bitmap.format() {
            BitmapFormat::Rgb | BitmapFormat::Yuv420p => (gl::RGB, bitmap.to_rgb()),
            BitmapFormat::Rgba | BitmapFormat::Yuva420p => (gl::RGBA, bitmap.to_rgba()),
        };

        unsafe {
            gl::BindTexture(gl::TEXTURE_2D, texture);
            gl::TexImage2D(
                gl::TEXTURE_2D,
                0,
                format as i32,
                bitmap.width() as i32,
                bitmap.height() as i32,
                0,
                format,
                gl::UNSIGNED_BYTE,
                bitmap.data().as_ptr() as *const gl::types::GLvoid,
            );
        }

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
        result.push("Renderer: GLES 2.0".to_string());

        let mut add_line = |name, idx: u32| {
            let ptr = unsafe { gl::GetString(idx as gl::types::GLenum) };
            let val = if !ptr.is_null() {
                Some(unsafe { std::ffi::CStr::from_ptr(ptr as *const std::os::raw::c_char).to_string_lossy().into_owned() })
            } else {
                None
            };

            result.push(format!(
                "{name}: {}",
                val.unwrap_or_else(|| "<unknown>".to_string())
            ))
        };

        add_line("Adapter Vendor", gl::VENDOR);
        add_line("Adapter Renderer", gl::RENDERER);
        add_line("Adapter Version", gl::VERSION);

        result.push(format!("Surface samples: {} x ", self.msaa_sample_count));
        result.push(format!(
            "Surface size: {} x {}",
            self.renderbuffer_width, self.renderbuffer_height
        ));

        Cow::Owned(result.join("\n"))
    }

    fn name(&self) -> &'static str {
        "gles2"
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
        unsafe {
            let mut texture: u32 = 0;
            gl::GenTextures(1, &mut texture);
            gl::BindTexture(gl::TEXTURE_2D, texture);

            // You must set the texture parameters for non-power-of-2 textures to function in WebGL1.
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);

            Ok(BitmapHandle(Arc::new(RegistryData {
                width,
                height,
                texture,
            })))
        }
    }
}

impl CommandHandler for GLES2RenderBackend {
    fn render_bitmap(
        &mut self,
        bitmap: BitmapHandle,
        transform: Transform,
        smoothing: bool,
        pixel_snapping: PixelSnapping,
    ) {
        self.set_stencil_state();
        let entry = as_registry_data(&bitmap);
        // Adjust the quad draw to use the target bitmap.
        let quad = &self.bitmap_quad_draws;
        let draw = &quad[0];
        let bitmap_matrix = if let DrawType::Bitmap(BitmapDraw { matrix, .. }) = &draw.draw_type {
            matrix
        } else {
            unreachable!()
        };

        // Scale the quad to the bitmap's dimensions.
        let mut matrix = transform.matrix;
        pixel_snapping.apply(&mut matrix);
        matrix *= Matrix::scale(entry.width as f32, entry.height as f32);

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

        let mult_color = transform.color_transform.mult_rgba_normalized();
        let add_color = transform.color_transform.add_rgba_normalized();

        self.bind_vertex_array(Some(draw.vao));

        let program = &self.bitmap_program;

        // Set common render state, while minimizing unnecessary state changes.
        // TODO: Using designated layout specifiers in WebGL2/OpenGL ES 3, we could guarantee that uniforms
        // are in the same location between shaders, and avoid changing them unless necessary.
        if program as *const ShaderProgram != self.active_program {
            unsafe {
                gl::UseProgram(program.program);
            }
            self.active_program = program as *const ShaderProgram;

            program.uniform_matrix4fv(ShaderUniform::ViewMatrix, &self.view_matrix);

            self.mult_color = None;
            self.add_color = None;
        }

        program.uniform_matrix4fv(ShaderUniform::WorldMatrix, &world_matrix);
        if Some(mult_color) != self.mult_color {
            program.uniform4fv(ShaderUniform::MultColor, &mult_color);
            self.mult_color = Some(mult_color);
        }
        if Some(add_color) != self.add_color {
            program.uniform4fv(ShaderUniform::AddColor, &add_color);
            self.add_color = Some(add_color);
        }

        program.uniform_matrix3fv(ShaderUniform::TextureMatrix, bitmap_matrix);

        // Bind texture.
        unsafe {
            gl::ActiveTexture(gl::TEXTURE0);
            gl::BindTexture(gl::TEXTURE_2D, entry.texture);
            
            program.uniform1i(ShaderUniform::BitmapTexture, 0);

            // Set texture parameters.
            let filter = if smoothing {
                gl::LINEAR as i32
            } else {
                gl::NEAREST as i32
            };
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, filter);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, filter);

            let wrap = gl::CLAMP_TO_EDGE as i32;
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, wrap);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, wrap);

            // Draw the triangles.
            gl::DrawElements(
                gl::TRIANGLE_FAN,
                draw.num_indices,
                gl::UNSIGNED_INT,
                0 as *const gl::types::GLvoid
            );
        }
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

            self.bind_vertex_array(Some(draw.vao));

            let program = match &draw.draw_type {
                DrawType::Color => &self.color_program,
                DrawType::Gradient(_) => &self.gradient_program,
                DrawType::Bitmap { .. } => &self.bitmap_program,
            };

            // Set common render state, while minimizing unnecessary state changes.
            // TODO: Using designated layout specifiers in WebGL2/OpenGL ES 3, we could guarantee that uniforms
            // are in the same location between shaders, and avoid changing them unless necessary.
            if program as *const ShaderProgram != self.active_program {
                unsafe {
                    gl::UseProgram(program.program);
                }
                self.active_program = program as *const ShaderProgram;

                program.uniform_matrix4fv(ShaderUniform::ViewMatrix, &self.view_matrix);

                self.mult_color = None;
                self.add_color = None;
            }

            program.uniform_matrix4fv(ShaderUniform::WorldMatrix, &world_matrix);
            if Some(mult_color) != self.mult_color {
                program.uniform4fv(ShaderUniform::MultColor, &mult_color);
                self.mult_color = Some(mult_color);
            }
            if Some(add_color) != self.add_color {
                program.uniform4fv(ShaderUniform::AddColor, &add_color);
                self.add_color = Some(add_color);
            }

            // Set shader specific uniforms.
            match &draw.draw_type {
                DrawType::Color => (),
                DrawType::Gradient(gradient) => {
                    program.uniform_matrix3fv(
                        ShaderUniform::TextureMatrix,
                        &gradient.matrix,
                    );
                    program.uniform1i(
                        ShaderUniform::GradientType,
                        gradient.gradient_type,
                    );
                    program.uniform1fv(ShaderUniform::GradientRatios, &gradient.ratios);
                    program.uniform4fv(
                        ShaderUniform::GradientColors,
                        bytemuck::cast_slice(&gradient.colors),
                    );
                    program.uniform1i(
                        ShaderUniform::GradientRepeatMode,
                        gradient.repeat_mode,
                    );
                    program.uniform1f(
                        ShaderUniform::GradientFocalPoint,
                        gradient.focal_point,
                    );
                    program.uniform1i(
                        ShaderUniform::GradientInterpolation,
                        (gradient.interpolation == swf::GradientInterpolation::LinearRgb) as i32,
                    );
                }
                DrawType::Bitmap(bitmap) => {
                    let texture = match &bitmap.handle {
                        Some(handle) => as_registry_data(handle).texture,
                        None => {
                            log::warn!("Tried to render a handleless bitmap");
                            continue;
                        }
                    };

                    program.uniform_matrix3fv(
                        ShaderUniform::TextureMatrix,
                        &bitmap.matrix,
                    );

                    // Bind texture.
                    unsafe {
                        gl::ActiveTexture(gl::TEXTURE0);
                        gl::BindTexture(gl::TEXTURE_2D, texture);
                        
                        program.uniform1i(ShaderUniform::BitmapTexture, 0);

                        // Set texture parameters.
                        let filter = if bitmap.is_smoothed {
                            gl::LINEAR as i32
                        } else {
                            gl::NEAREST as i32
                        };
                        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, filter);
                        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, filter);
                        let wrap = if bitmap.is_repeating {
                            gl::REPEAT as i32
                        } else {
                            gl::CLAMP_TO_EDGE as i32
                        };
                        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, wrap);
                        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, wrap);
                    }
                }
            }

            // Draw the triangles.
            unsafe {
                gl::DrawElements(
                    gl::TRIANGLES,
                    num_indices,
                    gl::UNSIGNED_INT,
                    0 as *const gl::types::GLvoid
                );
            }
        }
    }

    fn render_stage3d(&mut self, _bitmap: BitmapHandle, _transform: Transform) {
        panic!("Stage3D should not have been created on GLES2 backend")
    }

    fn draw_rect(&mut self, color: Color, matrix: Matrix) {
        self.draw_quad::<{ gl::TRIANGLE_FAN }, -1>(color, matrix)
    }

    fn draw_line(&mut self, color: Color, mut matrix: Matrix) {
        matrix.tx += Twips::HALF;
        matrix.ty += Twips::HALF;
        self.draw_quad::<{ gl::LINE_STRIP }, 2>(color, matrix)
    }

    fn draw_line_rect(&mut self, color: Color, mut matrix: Matrix) {
        matrix.tx += Twips::HALF;
        matrix.ty += Twips::HALF;
        self.draw_quad::<{ gl::LINE_LOOP }, -1>(color, matrix)
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

#[derive(Clone, Debug)]
struct Gradient {
    matrix: [[f32; 3]; 3],
    gradient_type: i32,
    ratios: [f32; MAX_GRADIENT_COLORS],
    colors: [[f32; 4]; MAX_GRADIENT_COLORS],
    repeat_mode: i32,
    focal_point: f32,
    interpolation: swf::GradientInterpolation,
}

impl Gradient {
    fn new(gradient: TessGradient, matrix: [[f32; 3]; 3]) -> Self {
        // TODO: Support more than MAX_GRADIENT_COLORS.
        let num_colors = gradient.records.len().min(MAX_GRADIENT_COLORS);
        let mut ratios = [0.0; MAX_GRADIENT_COLORS];
        let mut colors = [[0.0; 4]; MAX_GRADIENT_COLORS];
        for i in 0..num_colors {
            let record = &gradient.records[i];
            let mut color = [
                f32::from(record.color.r) / 255.0,
                f32::from(record.color.g) / 255.0,
                f32::from(record.color.b) / 255.0,
                f32::from(record.color.a) / 255.0,
            ];
            // Convert to linear color space if this is a linear-interpolated gradient.
            match gradient.interpolation {
                swf::GradientInterpolation::Rgb => {}
                swf::GradientInterpolation::LinearRgb => srgb_to_linear(&mut color),
            }

            colors[i] = color;
            ratios[i] = f32::from(record.ratio) / 255.0;
        }

        for i in num_colors..MAX_GRADIENT_COLORS {
            ratios[i] = ratios[i - 1];
            colors[i] = colors[i - 1];
        }

        Self {
            matrix,
            gradient_type: match gradient.gradient_type {
                GradientType::Linear => 0,
                GradientType::Radial => 1,
                GradientType::Focal => 2,
            },
            ratios,
            colors,
            repeat_mode: match gradient.repeat_mode {
                swf::GradientSpread::Pad => 0,
                swf::GradientSpread::Repeat => 1,
                swf::GradientSpread::Reflect => 2,
            },
            focal_point: gradient.focal_point.to_f32().clamp(-0.98, 0.98),
            interpolation: gradient.interpolation,
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

impl Drop for Mesh {
    fn drop(&mut self) {
        unsafe {
            for draw in &self.draws {
                gl::DeleteVertexArrays(1, &draw.vao);
            }
        }
    }
}

impl ShapeHandleImpl for Mesh {}

fn as_mesh(handle: &ShapeHandle) -> &Mesh {
    <dyn ShapeHandleImpl>::downcast_ref(&*handle.0).expect("Shape handle must be a WebGL ShapeData")
}

#[derive(Debug)]
struct Buffer {
    buffer: gl::types::GLuint,
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteBuffers(1, &self.buffer);
        }
    }
}

#[allow(dead_code)]
#[derive(Debug)]
struct Draw {
    draw_type: DrawType,
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    vao: gl::types::GLuint,
    num_indices: i32,
    num_mask_indices: i32,
}

#[derive(Debug)]
enum DrawType {
    Color,
    Gradient(Box<Gradient>),
    Bitmap(BitmapDraw),
}

struct MsaaBuffers {
    color_renderbuffer: gl::types::GLuint,
    stencil_renderbuffer: gl::types::GLuint,
    render_framebuffer: gl::types::GLuint,
    color_framebuffer: gl::types::GLuint,
    framebuffer_texture: gl::types::GLuint,
}

// Because the shaders are currently simple and few in number, we are using a
// straightforward shader model. We maintain an enum of every possible uniform,
// and each shader tries to grab the location of each uniform.
struct ShaderProgram {
    name: String,
    program: gl::types::GLuint,
    uniforms: [gl::types::GLint; NUM_UNIFORMS],
    vertex_position_location: u32,
    vertex_color_location: u32,
    num_vertex_attributes: u32,
}

// These should match the uniform names in the shaders.
const NUM_UNIFORMS: usize = 12;
const UNIFORM_NAMES: [&str; NUM_UNIFORMS] = [
    "world_matrix",
    "view_matrix",
    "mult_color",
    "add_color",
    "u_matrix",
    "u_gradient_type",
    "u_ratios",
    "u_colors",
    "u_repeat_mode",
    "u_focal_point",
    "u_interpolation",
    "u_texture",
];

enum ShaderUniform {
    WorldMatrix = 0,
    ViewMatrix,
    MultColor,
    AddColor,
    TextureMatrix,
    GradientType,
    GradientRatios,
    GradientColors,
    GradientRepeatMode,
    GradientFocalPoint,
    GradientInterpolation,
    BitmapTexture,
}

pub unsafe fn get_program_info_log(program: gl::types::GLuint) -> String {
    let mut length: gl::types::GLint = 0;
    gl::GetProgramiv(program, gl::INFO_LOG_LENGTH, &mut length);

    if length > 0 {
        let mut buffer = Vec::with_capacity(length as usize);
        buffer.set_len((length as usize) - 1);
        gl::GetProgramInfoLog(
            program,
            length,
            std::ptr::null_mut(),
            buffer.as_mut_ptr() as *mut gl::types::GLchar,
        );
        String::from_utf8_lossy(&buffer).into_owned()
    } else {
        String::new()
    }
}


pub unsafe fn get_shader_info_log(shader: gl::types::GLuint) -> String {
    let mut length: gl::types::GLint = 0;
    gl::GetShaderiv(shader, gl::INFO_LOG_LENGTH, &mut length);

    if length > 0 {
        let mut buffer = Vec::with_capacity(length as usize);
        buffer.set_len((length as usize) - 1);
        gl::GetShaderInfoLog(
            shader,
            length,
            std::ptr::null_mut(),
            buffer.as_mut_ptr() as *mut gl::types::GLchar,
        );

        String::from_utf8_lossy(&buffer).into_owned()
    } else {
        String::new()
    }
}

#[macro_export]
macro_rules! to_cstring {
    ($s:expr) => {{
        ::std::ffi::CString::new($s).expect("String contains a null byte").as_ptr()
    }};
}

impl ShaderProgram {
    fn new(
        name: &str,
        vertex_shader: gl::types::GLuint,
        fragment_shader: gl::types::GLuint,
    ) -> Result<Self, Error> {
        unsafe {
            let program = gl::CreateProgram();
            if program == 0 {
                return Err(Error::UnableToCreateProgram);
            }

            gl::AttachShader(program, vertex_shader);
            gl::AttachShader(program, fragment_shader);
            gl::LinkProgram(program);

            let mut link_status: gl::types::GLint = 0;
            gl::GetProgramiv(program, gl::LINK_STATUS, &mut link_status);
            if link_status != 1 {
                let msg = format!(
                    "Error linking shader program: {:?}",
                    get_program_info_log(program)
                );
                log::error!("{}", msg);
                return Err(Error::LinkingShaderProgram(msg));
            }

            // Find uniforms.
            let mut uniforms: [gl::types::GLint; NUM_UNIFORMS] = Default::default();
            for i in 0..NUM_UNIFORMS {
                uniforms[i] = gl::GetUniformLocation(program, to_cstring!(UNIFORM_NAMES[i]));
            }

            let vertex_position_location = gl::GetAttribLocation(program, to_cstring!("position")) as u32;
            let vertex_color_location = gl::GetAttribLocation(program, to_cstring!("color")) as u32;
            let num_vertex_attributes = if vertex_position_location != 0xffff_ffff {
                1
            } else {
                0
            } + if vertex_color_location != 0xffff_ffff {
                1
            } else {
                0
            };

            Ok(ShaderProgram {
                name: name.into(),
                program,
                uniforms,
                vertex_position_location,
                vertex_color_location,
                num_vertex_attributes,
            })
        }
    }

    fn uniform1f(&self, uniform: ShaderUniform, value: f32) {
        unsafe {
            gl::Uniform1f(self.uniforms[uniform as usize], value);
        }
    }

    fn uniform1fv(&self, uniform: ShaderUniform, values: &[f32]) {
        unsafe {
            gl::Uniform1fv(
                self.uniforms[uniform as usize],
                values.len() as i32,
                values.as_ptr()
            );
        }
    }

    fn uniform1i(&self, uniform: ShaderUniform, value: i32) {
        unsafe {
            gl::Uniform1i(self.uniforms[uniform as usize], value);
        }
    }

    fn uniform4fv(&self, uniform: ShaderUniform, values: &[f32]) {
        unsafe {
            gl::Uniform4fv(
                self.uniforms[uniform as usize],
                (values.len()/4) as i32,
                values.as_ptr()
            );
        }
    }

    fn uniform_matrix3fv(&self, uniform: ShaderUniform, values: &[[f32; 3]; 3]) {
        unsafe {
            gl::UniformMatrix3fv(
                self.uniforms[uniform as usize],
                1,
                0,
                values.as_ptr() as *const gl::types::GLfloat,
            );
        }
    }

    fn uniform_matrix4fv(&self, uniform: ShaderUniform, values: &[[f32; 4]; 4]) {
        unsafe {
            gl::UniformMatrix4fv(
                self.uniforms[uniform as usize],
                1,
                0,
                values.as_ptr() as *const gl::types::GLfloat,
            );
        }
    }
}

fn gl_check_error(error_msg: &'static str) -> Result<(), Error> {
    unsafe {
        match gl::GetError() {
            gl::NO_ERROR => Ok(()),
            error => Err(Error::GLError(error_msg, error)),
        }
    }
}

/// Converts an RGBA color from sRGB space to linear color space.
fn srgb_to_linear(color: &mut [f32; 4]) {
    for n in &mut color[..3] {
        *n = if *n <= 0.04045 {
            *n / 12.92
        } else {
            f32::powf((*n + 0.055) / 1.055, 2.4)
        };
    }
}
