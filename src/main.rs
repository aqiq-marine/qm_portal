use bytemuck::{Pod, Zeroable};
use serde::Deserialize;
use std::{
    env, fs,
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};
use wgpu::util::DeviceExt;
use wgpui::{prelude::*, *};

#[derive(Clone, Deserialize)]
struct Molecule {
    charge: i32,
    spin: i32,
    atoms: Vec<Atom>,
    #[serde(default)]
    bonds: Vec<Bond>,
}
#[derive(Clone, Deserialize)]
struct Atom {
    element: String,
    x: f32,
    y: f32,
    z: f32,
}
#[derive(Clone, Deserialize)]
struct Bond {
    from: usize,
    to: usize,
    order: u8,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    pos: [f32; 3],
    color: [f32; 3],
}
#[derive(Clone, Copy)]
struct Camera {
    yaw: f32,
    pitch: f32,
    zoom: f32,
    pan_x: f32,
    pan_y: f32,
}
impl Default for Camera {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.0,
            zoom: 0.55,
            pan_x: 0.0,
            pan_y: 0.0,
        }
    }
}

fn color(e: &str) -> [f32; 3] {
    match e {
        "H" => [0.92, 0.95, 1.0],
        "C" => [0.18, 0.2, 0.23],
        "N" => [0.15, 0.3, 0.95],
        "O" => [0.95, 0.12, 0.1],
        "F" => [0.55, 0.95, 0.55],
        _ => [0.7, 0.7, 0.75],
    }
}
fn mesh(m: &Molecule) -> Vec<Vertex> {
    let mut v = Vec::new();
    for a in &m.atoms {
        let r = if a.element == "H" { 0.16 } else { 0.25 };
        let c = color(&a.element);
        let n = 12;
        let rings = 8;
        for j in 0..rings {
            let p0 = std::f32::consts::PI * j as f32 / rings as f32;
            let p1 = std::f32::consts::PI * (j + 1) as f32 / rings as f32;
            for i in 0..n {
                let t0 = 2.0 * std::f32::consts::PI * i as f32 / n as f32;
                let t1 = 2.0 * std::f32::consts::PI * (i + 1) as f32 / n as f32;
                let q = |p: f32, t: f32| {
                    [
                        a.x + r * p.sin() * t.cos(),
                        a.y + r * p.cos(),
                        a.z + r * p.sin() * t.sin(),
                    ]
                };
                let (a0, a1, b0, b1) = (q(p0, t0), q(p0, t1), q(p1, t0), q(p1, t1));
                v.extend_from_slice(&[
                    Vertex { pos: a0, color: c },
                    Vertex { pos: b0, color: c },
                    Vertex { pos: b1, color: c },
                    Vertex { pos: a0, color: c },
                    Vertex { pos: b1, color: c },
                    Vertex { pos: a1, color: c },
                ]);
            }
        }
    }
    for b in &m.bonds {
        if let (Some(a), Some(d)) = (m.atoms.get(b.from), m.atoms.get(b.to)) {
            let c = [0.65, 0.68, 0.72];
            v.extend_from_slice(&[
                Vertex {
                    pos: [a.x, a.y, a.z],
                    color: c,
                },
                Vertex {
                    pos: [d.x, d.y, d.z],
                    color: c,
                },
            ]);
        }
    }
    v
}

struct View {
    surface: WgpuSurfaceHandle,
    molecule: Molecule,
    camera: Arc<Mutex<Camera>>,
    last_pointer: Option<Point<Pixels>>,
}
impl Render for View {
    fn render(&mut self, w: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        w.request_animation_frame();
        div().size_full().flex().flex_col().bg(rgb(0x101318)).child(div().px_6().py_4().text_color(rgb(0xe7ebf2)).child(format!("Molecule · {} atoms · {} bonds · charge {} · spin {} · left drag: rotate · right drag: pan · wheel: zoom",self.molecule.atoms.len(),self.molecule.bonds.len(),self.molecule.charge,self.molecule.spin))).child(div().flex_1().m_4().relative().overflow_hidden().on_scroll_wheel(cx.listener(|this,event: &ScrollWheelEvent,_,cx| { let amount=match event.delta { ScrollDelta::Pixels(delta)=>f32::from(delta.y)/80.0, ScrollDelta::Lines(delta)=>delta.y }; let mut c=this.camera.lock().unwrap(); c.zoom=(c.zoom*(1.0-amount*0.12)).clamp(0.15,3.0); cx.notify(); })).on_mouse_move(cx.listener(|this,event: &MouseMoveEvent,_,cx| { if let Some(last)=this.last_pointer { let dx=f32::from(event.position.x)-f32::from(last.x); let dy=f32::from(event.position.y)-f32::from(last.y); let mut c=this.camera.lock().unwrap(); match event.pressed_button { Some(MouseButton::Left)=>{c.yaw+=dx*0.012;c.pitch=(c.pitch+dy*0.012).clamp(-1.5,1.5);}, Some(MouseButton::Right)=>{c.pan_x+=dx*0.002;c.pan_y-=dy*0.002;}, _=>{} } } this.last_pointer=Some(event.position); cx.notify(); })).child(wgpu_surface(self.surface.clone()).absolute().inset_0()))
    }
}

fn render_loop(surface: WgpuSurfaceHandle, molecule: Molecule, camera: Arc<Mutex<Camera>>) {
    while surface.back_buffer_view().is_none() {
        thread::sleep(Duration::from_millis(10));
    }
    let device = surface.device();
    let queue = surface.queue();
    let shader=device.create_shader_module(wgpu::ShaderModuleDescriptor{label:Some("molecule"),source:wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(r#"struct U{yaw:f32,pitch:f32,zoom:f32,_pad:f32,pan_x:f32,pan_y:f32,_pad2:vec2<f32>,};@group(0)@binding(0)var<uniform>u:U;struct I{@location(0)p:vec3<f32>,@location(1)c:vec3<f32>};struct O{@builtin(position)p:vec4<f32>,@location(0)c:vec3<f32>};@vertex fn vs(i:I)->O{let cy=cos(u.yaw);let sy=sin(u.yaw);let x=i.p.x*cy-i.p.z*sy;let z=i.p.x*sy+i.p.z*cy;let cp=cos(u.pitch);let sp=sin(u.pitch);let y=i.p.y*cp-z*sp;let rz=i.p.y*sp+z*cp;var o:O;o.p=vec4<f32>((x+u.pan_x)*u.zoom,(y+u.pan_y)*u.zoom,rz*.12+0.5,1);o.c=i.c;return o;}@fragment fn fs(i:O)->@location(0)vec4<f32>{return vec4<f32>(i.c,1)}"#))});
    let verts = mesh(&molecule);
    let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("molecule vertices"),
        contents: bytemuck::cast_slice(&verts),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let ub = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("camera"),
        size: 32,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: ub.as_entire_binding(),
        }],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bl)],
        immediate_size: 0,
    });
    let pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pl),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Vertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 12,
                        shader_location: 1,
                    },
                ],
            })],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });
    loop {
        surface.wait_for_present();
        let Some(view) = surface.back_buffer_view() else {
            continue;
        };
        let c = *camera.lock().unwrap();
        queue.write_buffer(
            &ub,
            0,
            bytemuck::cast_slice(&[c.yaw, c.pitch, c.zoom, 0.0, c.pan_x, c.pan_y, 0.0, 0.0]),
        );
        let mut e = device.create_command_encoder(&Default::default());
        {
            let mut p = e.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.04,
                            g: 0.05,
                            b: 0.07,
                            a: 1.,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                multiview_mask: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            p.set_pipeline(&pipe);
            p.set_bind_group(0, &bg, &[]);
            p.set_vertex_buffer(0, vb.slice(..));
            p.draw(0..verts.len() as u32, 0..1);
        }
        queue.submit(Some(e.finish()));
        surface.present();
    }
}

fn main() {
    let path = env::args().nth(1).unwrap_or_else(|| "molecule.json".into());
    let text = fs::read_to_string(path).expect("molecule JSON");
    let molecule: Molecule = serde_json::from_str(&text).expect("valid molecule JSON");
    Application::new().run(move |cx| {
        cx.open_window(WindowOptions::default(), move |w, cx| {
            let surface = w
                .create_wgpu_surface(1200, 800, wgpu::TextureFormat::Rgba8UnormSrgb)
                .expect("WgpuSurface unavailable");
            let draw = surface.clone();
            let data = molecule.clone();
            let camera = Arc::new(Mutex::new(Camera::default()));
            let draw_camera = camera.clone();
            thread::spawn(move || render_loop(draw, data, draw_camera));
            cx.new(|_| View {
                surface,
                molecule,
                camera,
                last_pointer: None,
            })
        })
        .unwrap();
    });
}
