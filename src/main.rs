#![recursion_limit = "256"]
use bytemuck::{Pod, Zeroable};
use serde::Deserialize;
use std::{
    env, fs,
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};
use editor::Camera;
use wgpu::util::DeviceExt;
use wgpui::{prelude::*, *};

mod editor;

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
    #[serde(default)]
    id: u64,
    element: String,
    x: f32,
    y: f32,
    z: f32,
}
#[derive(Clone, Deserialize, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum BondType {
    None,
    Single,
    Double,
    Triple,
    Aromatic,
}

#[derive(Clone, Deserialize)]
struct Bond {
    #[serde(default)]
    id: u64,
    from: usize,
    to: usize,
    order: BondType,
}

fn normalize_ids(molecule: &mut Molecule) {
    for (index, atom) in molecule.atoms.iter_mut().enumerate() {
        if atom.id == 0 {
            atom.id = (index + 1) as u64;
        }
    }
    for (index, bond) in molecule.bonds.iter_mut().enumerate() {
        if bond.id == 0 {
            bond.id = (index + 1) as u64;
        }
    }
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    pos: [f32; 3],
    color: [f32; 3],
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

fn atom_radius(atom: &Atom) -> f32 {
    if atom.element == "H" { 0.16 } else { 0.25 }
}

fn mesh(m: &Molecule, selected: &[u64]) -> Vec<Vertex> {
    let mut v = Vec::new();
    for a in &m.atoms {
        let r = atom_radius(a);
        let c = if selected.contains(&a.id) {
            [1.0, 0.9, 0.1]
        } else {
            color(&a.element)
        };
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
            let p1 = [a.x, a.y, a.z];
            let p2 = [d.x, d.y, d.z];
            let c = [0.65, 0.68, 0.72];
            let dx = p2[0] - p1[0];
            let dy = p2[1] - p1[1];
            let dz = p2[2] - p1[2];
            let len = (dx * dx + dy * dy + dz * dz).sqrt();
            if len > 1e-5 {
                let dir = [dx / len, dy / len, dz / len];
                let arbitrary = if dir[1].abs() < 0.9 {
                    [0.0, 1.0, 0.0]
                } else {
                    [1.0, 0.0, 0.0]
                };
                let u1_x = dir[1] * arbitrary[2] - dir[2] * arbitrary[1];
                let u1_y = dir[2] * arbitrary[0] - dir[0] * arbitrary[2];
                let u1_z = dir[0] * arbitrary[1] - dir[1] * arbitrary[0];
                let u1_len = (u1_x * u1_x + u1_y * u1_y + u1_z * u1_z).sqrt();
                let u1 = if u1_len > 1e-5 {
                    [u1_x / u1_len, u1_y / u1_len, u1_z / u1_len]
                } else {
                    [1.0, 0.0, 0.0]
                };

                match b.order {
                    BondType::None => {}
                    BondType::Single => {
                        add_cylinder(&mut v, p1, p2, 0.08, c, [0.0, 0.0, 0.0]);
                    }
                    BondType::Double => {
                        let offset = [u1[0] * 0.06, u1[1] * 0.06, u1[2] * 0.06];
                        add_cylinder(&mut v, p1, p2, 0.06, c, offset);
                        add_cylinder(
                            &mut v,
                            p1,
                            p2,
                            0.06,
                            c,
                            [-offset[0], -offset[1], -offset[2]],
                        );
                    }
                    BondType::Triple => {
                        add_cylinder(&mut v, p1, p2, 0.05, c, [0.0, 0.0, 0.0]);
                        let offset = [u1[0] * 0.08, u1[1] * 0.08, u1[2] * 0.08];
                        add_cylinder(&mut v, p1, p2, 0.05, c, offset);
                        add_cylinder(
                            &mut v,
                            p1,
                            p2,
                            0.05,
                            c,
                            [-offset[0], -offset[1], -offset[2]],
                        );
                    }
                    BondType::Aromatic => {
                        let offset = [u1[0] * 0.06, u1[1] * 0.06, u1[2] * 0.06];
                        add_cylinder(&mut v, p1, p2, 0.06, c, offset);
                        add_cylinder(
                            &mut v,
                            p1,
                            p2,
                            0.06,
                            c,
                            [-offset[0], -offset[1], -offset[2]],
                        );
                    }
                }
            }
        }
    }
    v
}

fn add_cylinder(
    v: &mut Vec<Vertex>,
    p1: [f32; 3],
    p2: [f32; 3],
    radius: f32,
    color: [f32; 3],
    offset: [f32; 3],
) {
    let dx = p2[0] - p1[0];
    let dy = p2[1] - p1[1];
    let dz = p2[2] - p1[2];
    let len = (dx * dx + dy * dy + dz * dz).sqrt();
    if len < 1e-5 {
        return;
    }
    let dir = [dx / len, dy / len, dz / len];

    let p1 = [p1[0] + offset[0], p1[1] + offset[1], p1[2] + offset[2]];
    let p2 = [p2[0] + offset[0], p2[1] + offset[1], p2[2] + offset[2]];

    let arbitrary = if dir[1].abs() < 0.9 {
        [0.0, 1.0, 0.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let u1_x = dir[1] * arbitrary[2] - dir[2] * arbitrary[1];
    let u1_y = dir[2] * arbitrary[0] - dir[0] * arbitrary[2];
    let u1_z = dir[0] * arbitrary[1] - dir[1] * arbitrary[0];
    let u1_len = (u1_x * u1_x + u1_y * u1_y + u1_z * u1_z).sqrt();
    if u1_len < 1e-5 {
        return;
    }
    let u1 = [u1_x / u1_len, u1_y / u1_len, u1_z / u1_len];
    let u2 = [
        dir[1] * u1[2] - dir[2] * u1[1],
        dir[2] * u1[0] - dir[0] * u1[2],
        dir[0] * u1[1] - dir[1] * u1[0],
    ];

    let segments = 8;
    let rings = 2;
    for j in 0..rings {
        let t0 = j as f32 / rings as f32;
        let t1 = (j + 1) as f32 / rings as f32;
        let c0_center = [
            p1[0] + (p2[0] - p1[0]) * t0,
            p1[1] + (p2[1] - p1[1]) * t0,
            p1[2] + (p2[2] - p1[2]) * t0,
        ];
        let c1_center = [
            p1[0] + (p2[0] - p1[0]) * t1,
            p1[1] + (p2[1] - p1[1]) * t1,
            p1[2] + (p2[2] - p1[2]) * t1,
        ];

        for i in 0..segments {
            let theta0 = 2.0 * std::f32::consts::PI * i as f32 / segments as f32;
            let theta1 = 2.0 * std::f32::consts::PI * (i + 1) as f32 / segments as f32;

            let get_pos = |center: [f32; 3], theta: f32| {
                let r_cos = radius * theta.cos();
                let r_sin = radius * theta.sin();
                [
                    center[0] + r_cos * u1[0] + r_sin * u2[0],
                    center[1] + r_cos * u1[1] + r_sin * u2[1],
                    center[2] + r_cos * u1[2] + r_sin * u2[2],
                ]
            };

            let v00 = get_pos(c0_center, theta0);
            let v01 = get_pos(c0_center, theta1);
            let v10 = get_pos(c1_center, theta0);
            let v11 = get_pos(c1_center, theta1);

            v.extend_from_slice(&[
                Vertex { pos: v00, color },
                Vertex { pos: v10, color },
                Vertex { pos: v11, color },
                Vertex { pos: v00, color },
                Vertex { pos: v11, color },
                Vertex { pos: v01, color },
            ]);
        }
    }
}

type Viewport = editor::PickViewport;

struct MoleculeSurface {
    handle: WgpuSurfaceHandle,
    viewport: Arc<Mutex<Viewport>>,
}

impl MoleculeSurface {
    fn new(handle: WgpuSurfaceHandle, viewport: Arc<Mutex<Viewport>>) -> Self {
        Self { handle, viewport }
    }
}

impl IntoElement for MoleculeSurface {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Element for MoleculeSurface {
    type RequestLayoutState = Style;
    type PrepaintState = ();

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static std::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let mut surface = wgpu_surface(self.handle.clone()).absolute().inset_0();
        surface.request_layout(id, inspector_id, window, cx)
    }

    fn prepaint(
        &mut self,
        id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        request_layout: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) -> Self::PrepaintState {
        *self.viewport.lock().unwrap() = Viewport {
            x: f32::from(bounds.origin.x),
            y: f32::from(bounds.origin.y),
            width: f32::from(bounds.size.width),
            height: f32::from(bounds.size.height),
        };

        let mut surface = wgpu_surface(self.handle.clone()).absolute().inset_0();
        surface.prepaint(id, inspector_id, bounds, request_layout, window, cx)
    }

    fn paint(
        &mut self,
        id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        style: &mut Self::RequestLayoutState,
        prepaint: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        let mut surface = wgpu_surface(self.handle.clone()).absolute().inset_0();
        surface.paint(id, inspector_id, bounds, style, prepaint, window, cx);
    }
}

struct View {
    surface: WgpuSurfaceHandle,
    molecule: Arc<Mutex<Molecule>>,
    editor: editor::EditorState,
    selected: Vec<u64>,
    selected_arc: Arc<Mutex<Vec<u64>>>,
    camera: Arc<Mutex<Camera>>,
    last_pointer: Option<Point<Pixels>>,
    edit_mode: EditMode,
    numeric_buffer: String,
    slider_last_x: Option<f32>,
    slider_value: Option<f64>,
    fragment_mode: bool,
    debug_mode: bool,
    debug_compact: bool,
    viewport: Arc<Mutex<Viewport>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EditMode {
    Select,
    BondLength,
    BondAngle,
    Dihedral,
}

impl EditMode {
    fn label(self) -> &'static str {
        match self {
            Self::Select => "select",
            Self::BondLength => "bond length (L)",
            Self::BondAngle => "bond angle (A)",
            Self::Dihedral => "dihedral (D)",
        }
    }
}

impl View {
    fn current_viewport(&self) -> Viewport {
        *self.viewport.lock().unwrap()
    }

    fn pick_atom(&self, position: Point<Pixels>) -> Option<u64> {
        let viewport = self.current_viewport();
        let camera = *self.camera.lock().unwrap();
        let molecule = self.molecule.lock().unwrap();

        editor::pick_atom_at(
            &molecule.atoms,
            camera,
            viewport,
            f32::from(position.x),
            f32::from(position.y),
        )
    }

    fn update_selection(&mut self, new_selected: Vec<u64>) {
        self.selected = new_selected;
        self.slider_last_x = None;
        self.slider_value = None;
        *self.selected_arc.lock().unwrap() = self.selected.clone();
    }

    fn handle_click(&mut self, event: &MouseDownEvent, cx: &mut Context<Self>) {
        let position = event.position;
        let Some(atom_id) = self.pick_atom(position) else {
            self.update_selection(Vec::new());
            cx.notify();
            return;
        };
        let mut new_sel = self.selected.clone();
        if let Some(index) = new_sel.iter().position(|id| *id == atom_id) {
            new_sel.remove(index);
        } else {
            if new_sel.len() >= 4 {
                new_sel = vec![atom_id];
            } else {
                new_sel.push(atom_id);
            }
        }
        self.update_selection(new_sel);
        cx.notify();
    }

    fn handle_key(&mut self, event: &KeyDownEvent, cx: &mut Context<Self>) {
        let key = event.keystroke.key.as_str();
        if event.keystroke.modifiers.control {
            let changed = match key {
                "z" => self.editor.undo(&mut self.molecule.lock().unwrap()),
                "y" => self.editor.redo(&mut self.molecule.lock().unwrap()),
                _ => false,
            };
            if changed {
                cx.notify();
            }
            return;
        }

        match key {
            "b" => self.cycle_selected_bond(),
            "g" => self.fragment_mode = !self.fragment_mode,
            "l" => {
                self.edit_mode = EditMode::BondLength;
                self.numeric_buffer.clear();
            }
            "a" => {
                self.edit_mode = EditMode::BondAngle;
                self.numeric_buffer.clear();
            }
            "d" => {
                self.edit_mode = EditMode::Dihedral;
                self.numeric_buffer.clear();
            }
            "escape" => {
                self.edit_mode = EditMode::Select;
                self.numeric_buffer.clear();
            }
            "backspace" => {
                self.numeric_buffer.pop();
            }
            "enter" => self.apply_numeric_edit(),
            _ if self.edit_mode != EditMode::Select
                && (key.chars().all(|c| c.is_ascii_digit()) || key == "." || key == "-") =>
            {
                self.numeric_buffer.push_str(key);
            }
            _ => return,
        }
        cx.notify();
    }

    fn apply_numeric_edit(&mut self) {
        let Ok(value) = self.numeric_buffer.parse::<f64>() else {
            return;
        };
        let command = match self.edit_mode {
            EditMode::BondLength if self.selected.len() == 2 => {
                Some(editor::EditCommand::SetBondLength {
                    from: self.selected[0],
                    to: self.selected[1],
                    length: value,
                    move_selection: self.move_selection(),
                })
            }
            EditMode::BondAngle if self.selected.len() == 3 => {
                Some(editor::EditCommand::SetBondAngle {
                    a: self.selected[0],
                    center: self.selected[1],
                    c: self.selected[2],
                    angle: value.to_radians(),
                    move_selection: self.move_selection(),
                })
            }
            EditMode::Dihedral if self.selected.len() == 4 => {
                Some(editor::EditCommand::SetDihedral {
                    a: self.selected[0],
                    b: self.selected[1],
                    c: self.selected[2],
                    d: self.selected[3],
                    angle: value.to_radians(),
                })
            }
            _ => None,
        };
        if let Some(command) = command {
            let _ = self
                .editor
                .apply(&mut self.molecule.lock().unwrap(), command);
        }
        self.numeric_buffer.clear();
    }

    fn move_selection(&self) -> editor::MoveSelection {
        if self.fragment_mode {
            editor::MoveSelection::ConnectedFragment
        } else {
            editor::MoveSelection::OneAtom
        }
    }

    fn cycle_selected_bond(&mut self) {
        if self.selected.len() != 2 {
            return;
        }
        let (bond_id, next_order) = {
            let molecule = self.molecule.lock().unwrap();
            let Some(bond) = molecule.bonds.iter().find(|bond| {
                let Some(from) = molecule.atoms.get(bond.from) else {
                    return false;
                };
                let Some(to) = molecule.atoms.get(bond.to) else {
                    return false;
                };
                (from.id == self.selected[0] && to.id == self.selected[1])
                    || (from.id == self.selected[1] && to.id == self.selected[0])
            }) else {
                return;
            };
            let next = match bond.order {
                BondType::None => BondType::Single,
                BondType::Single => BondType::Double,
                BondType::Double => BondType::Triple,
                BondType::Triple => BondType::Aromatic,
                BondType::Aromatic => BondType::None,
            };
            (bond.id, next)
        };
        let _ = self.editor.apply(
            &mut self.molecule.lock().unwrap(),
            editor::EditCommand::SetBondOrder {
                bond_id,
                order: next_order,
            },
        );
    }

    fn set_edit_mode(&mut self, mode: EditMode, cx: &mut Context<Self>) {
        let previous_mode = self.edit_mode;
        match mode {
            EditMode::BondLength if self.selected.len() == 2 => self.edit_mode = mode,
            EditMode::BondAngle if self.selected.len() == 3 => self.edit_mode = mode,
            EditMode::Dihedral if self.selected.len() == 4 => self.edit_mode = mode,
            EditMode::Select => self.edit_mode = mode,
            _ => {}
        }
        if self.edit_mode != previous_mode {
            self.slider_last_x = None;
            self.slider_value = None;
        }
        cx.notify();
    }

    fn apply_value(&mut self, value: f64) {
        let molecule = self.molecule.lock().unwrap();
        let command = match self.edit_mode {
            EditMode::BondLength if self.selected.len() == 2 => {
                Some(editor::EditCommand::SetBondLength {
                    from: self.selected[0],
                    to: self.selected[1],
                    length: value,
                    move_selection: self.move_selection(),
                })
            }
            EditMode::BondAngle if self.selected.len() == 3 => {
                Some(editor::EditCommand::SetBondAngle {
                    a: self.selected[0],
                    center: self.selected[1],
                    c: self.selected[2],
                    angle: value.to_radians(),
                    move_selection: self.move_selection(),
                })
            }
            EditMode::Dihedral if self.selected.len() == 4 => {
                Some(editor::EditCommand::SetDihedral {
                    a: self.selected[0],
                    b: self.selected[1],
                    c: self.selected[2],
                    d: self.selected[3],
                    angle: value.to_radians(),
                })
            }
            _ => None,
        };
        drop(molecule);
        if let Some(command) = command {
            let _ = self
                .editor
                .apply(&mut self.molecule.lock().unwrap(), command);
        }
    }

    fn slider_range(&self) -> Option<(f64, f64)> {
        match self.edit_mode {
            EditMode::BondLength if self.selected.len() == 2 => Some((0.5, 4.0)),
            EditMode::BondAngle if self.selected.len() == 3 => Some((0.0, 180.0)),
            EditMode::Dihedral if self.selected.len() == 4 => Some((-180.0, 180.0)),
            _ => None,
        }
    }

    fn begin_slider(&mut self, x: f32) {
        self.slider_last_x = Some(x);
        if self.slider_value.is_none() {
            if let (Some(value), Some((min, max))) = (self.current_edit_value(), self.slider_range()) {
                self.slider_value = Some(value.clamp(min, max));
            }
        }
    }

    fn drag_slider(&mut self, x: f32) {
        let Some(last_x) = self.slider_last_x else {
            return;
        };
        let Some((min, max)) = self.slider_range() else {
            return;
        };
        self.slider_last_x = Some(x);
        let value_per_pixel = (max - min) / 280.0;
        let Some(current) = self.slider_value.or_else(|| self.current_edit_value()) else {
            return;
        };
        let next = (current + f64::from(x - last_x) * value_per_pixel).clamp(min, max);
        self.slider_value = Some(next);
        self.apply_value(next);
    }

    fn current_edit_value(&self) -> Option<f64> {
        let molecule = self.molecule.lock().unwrap();
        match self.edit_mode {
            EditMode::BondLength if self.selected.len() == 2 => {
                editor::distance(&molecule, self.selected[0], self.selected[1]).ok()
            }
            EditMode::BondAngle if self.selected.len() == 3 => editor::angle(
                &molecule,
                self.selected[0],
                self.selected[1],
                self.selected[2],
            )
            .ok()
            .map(f64::to_degrees),
            EditMode::Dihedral if self.selected.len() == 4 => editor::dihedral(
                &molecule,
                self.selected[0],
                self.selected[1],
                self.selected[2],
                self.selected[3],
            )
            .ok()
            .map(f64::to_degrees),
            _ => None,
        }
    }

    fn slider_knob_x(&self) -> Option<f32> {
        let (min, max) = self.slider_range()?;
        let value = self
            .slider_value
            .or_else(|| self.current_edit_value())?
            .clamp(min, max);
        Some(20.0 + ((value - min) / (max - min)) as f32 * 280.0)
    }

    fn slider_element(&self, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .relative()
            .w_80()
            .h_6()
            .bg(rgb(0x374151))
            .on_mouse_move(cx.listener(|this, event: &MouseMoveEvent, _, cx| {
                if event.pressed_button == Some(MouseButton::Left) {
                    this.drag_slider(f32::from(event.position.x));
                    cx.notify();
                }
            }))
            .child(
                div()
                    .absolute()
                    .left(px(20.0))
                    .right(px(20.0))
                    .top(px(11.0))
                    .h(px(4.0))
                    .bg(rgb(0x9ca3af)),
            )
            .child(
                div()
                    .absolute()
                    .left(px(self.slider_knob_x().unwrap_or(20.0) - 7.0))
                    .top(px(5.0))
                    .w(px(14.0))
                    .h(px(16.0))
                    .bg(rgb(0x60a5fa))
                    .border_1()
                    .border_color(rgb(0xdbeafe))
                    .on_mouse_down(MouseButton::Left, cx.listener(|this, event: &MouseDownEvent, _, cx| {
                        this.begin_slider(f32::from(event.position.x));
                        cx.notify();
                    })),
            )
    }
}
impl Render for View {
    fn render(&mut self, w: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        w.request_animation_frame();
        let molecule = self.molecule.lock().unwrap();
        let current_val_str = match self.edit_mode {
            EditMode::BondLength if self.selected.len() == 2 => {
                editor::distance(&molecule, self.selected[0], self.selected[1])
                    .map(|v| format!("{:.3} Å", v))
                    .unwrap_or_else(|_| "-".into())
            }
            EditMode::BondAngle if self.selected.len() == 3 => {
                editor::angle(&molecule, self.selected[0], self.selected[1], self.selected[2])
                    .map(|v| format!("{:.1}°", v.to_degrees()))
                    .unwrap_or_else(|_| "-".into())
            }
            EditMode::Dihedral if self.selected.len() == 4 => {
                editor::dihedral(&molecule, self.selected[0], self.selected[1], self.selected[2], self.selected[3])
                    .map(|v| format!("{:.1}°", v.to_degrees()))
                    .unwrap_or_else(|_| "-".into())
            }
            _ => "Select atoms (2 for length, 3 for angle, 4 for dihedral)".into(),
        };
        let summary = format!("Molecule · {} atoms · {} bonds · charge {} · spin {} · selected {:?} · mode: {} · value: {} · fragment: {} · B: bond order · G: fragment move · L/A/D then number + Enter · Ctrl+Z/Y: undo/redo", molecule.atoms.len(), molecule.bonds.len(), molecule.charge, molecule.spin, self.selected, self.edit_mode.label(), self.numeric_buffer, self.fragment_mode);
        drop(molecule);
        div()
            .size_full()
            .flex()
            .flex_col()
            .bg(rgb(0x101318))
            .on_key_down(
                cx.listener(|this, event: &KeyDownEvent, _, cx| this.handle_key(event, cx)),
            )
            .child(div().px_6().py_3().text_color(rgb(0xe7ebf2)).child(summary))
            .child(
                div()
                    .px_6()
                    .py_2()
                    .flex()
                    .flex_col()
                    .gap_2()
                    .text_color(rgb(0xe7ebf2))
                    .child(
                        div()
                            .flex()
                            .gap_2()
                            .child(
                                div()
                                    .px_3()
                                    .py_1()
                                    .bg(if self.selected.len() == 2 { rgb(0x2563eb) } else { rgb(0x1e222b) })
                                    .text_color(if self.selected.len() == 2 { rgb(0xffffff) } else { rgb(0x6b7280) })
                                    .on_mouse_down(MouseButton::Left, cx.listener(|this, _, _, cx| {
                                        if this.selected.len() == 2 {
                                            this.set_edit_mode(EditMode::BondLength, cx);
                                        }
                                    }))
                                    .child("Bond Length")
                            )
                            .child(
                                div()
                                    .px_3()
                                    .py_1()
                                    .bg(if self.selected.len() == 3 { rgb(0x2563eb) } else { rgb(0x1e222b) })
                                    .text_color(if self.selected.len() == 3 { rgb(0xffffff) } else { rgb(0x6b7280) })
                                    .on_mouse_down(MouseButton::Left, cx.listener(|this, _, _, cx| {
                                        if this.selected.len() == 3 {
                                            this.set_edit_mode(EditMode::BondAngle, cx);
                                        }
                                    }))
                                    .child("Bond Angle")
                            )
                            .child(
                                div()
                                    .px_3()
                                    .py_1()
                                    .bg(if self.selected.len() == 4 { rgb(0x2563eb) } else { rgb(0x1e222b) })
                                    .text_color(if self.selected.len() == 4 { rgb(0xffffff) } else { rgb(0x6b7280) })
                                    .on_mouse_down(MouseButton::Left, cx.listener(|this, _, _, cx| {
                                        if this.selected.len() == 4 {
                                            this.set_edit_mode(EditMode::Dihedral, cx);
                                        }
                                    }))
                                    .child("Dihedral Angle")
                            )
                            .child(
                                div()
                                    .px_3()
                                    .py_1()
                                    .bg(if self.selected.len() == 2 { rgb(0x059669) } else { rgb(0x1e222b) })
                                    .text_color(if self.selected.len() == 2 { rgb(0xffffff) } else { rgb(0x6b7280) })
                                    .on_mouse_down(MouseButton::Left, cx.listener(|this, _, _, cx| {
                                        if this.selected.len() == 2 {
                                            this.cycle_selected_bond();
                                            cx.notify();
                                        }
                                    }))
                                    .child("Cycle Bond")
                            )
                            .child(
                                div()
                                    .px_3()
                                    .py_1()
                                    .bg(if self.fragment_mode { rgb(0xd97706) } else { rgb(0x1e222b) })
                                    .text_color(rgb(0xffffff))
                                    .on_mouse_down(MouseButton::Left, cx.listener(|this, _, _, cx| {
                                        this.fragment_mode = !this.fragment_mode;
                                        cx.notify();
                                    }))
                                    .child(format!("Fragment: {}", if self.fragment_mode { "ON" } else { "OFF" }))
                            )
                            .child(
                                div()
                                    .px_3()
                                    .py_1()
                                    .bg(rgb(0x374151))
                                    .text_color(rgb(0xffffff))
                                    .on_mouse_down(MouseButton::Left, cx.listener(|this, _, _, cx| {
                                        if this.editor.undo(&mut this.molecule.lock().unwrap()) {
                                            cx.notify();
                                        }
                                    }))
                                    .child("Undo")
                            )
                            .child(
                                div()
                                    .px_3()
                                    .py_1()
                                    .bg(rgb(0x374151))
                                    .text_color(rgb(0xffffff))
                                    .on_mouse_down(MouseButton::Left, cx.listener(|this, _, _, cx| {
                                        if this.editor.redo(&mut this.molecule.lock().unwrap()) {
                                            cx.notify();
                                        }
                                    }))
                                    .child("Redo")
                            )
                            .child(
                                div()
                                    .px_3()
                                    .py_1()
                                    .bg(if self.debug_mode { rgb(0x7c3aed) } else { rgb(0x1e222b) })
                                    .text_color(rgb(0xffffff))
                                    .on_mouse_down(MouseButton::Left, cx.listener(|this, _, _, cx| {
                                        this.debug_mode = !this.debug_mode;
                                        cx.notify();
                                    }))
                                    .child(format!("Debug: {}", if self.debug_mode { "ON" } else { "OFF" }))
                            )
                    )
                    .child(match self.edit_mode {
                        EditMode::BondLength if self.selected.len() == 2 => {
                            div().flex().gap_4().items_center()
                                .child(format!("Bond Length: {}", current_val_str))
                                .child(self.slider_element(cx))
                        }
                        EditMode::BondAngle if self.selected.len() == 3 => {
                            div().flex().gap_4().items_center()
                                .child(format!("Bond Angle: {}", current_val_str))
                                .child(self.slider_element(cx))
                        }
                        EditMode::Dihedral if self.selected.len() == 4 => {
                            div().flex().gap_4().items_center()
                                .child(format!("Dihedral Angle: {}", current_val_str))
                                .child(self.slider_element(cx))
                        }
                        _ => div().child(format!("Mode: {} | {}", self.edit_mode.label(), current_val_str)),
                    })
            )
            .child(
                {
                    let debug_mode = self.debug_mode;
                    let viewport = self.current_viewport();
                    let camera = {
                        let mut c = self.camera.lock().unwrap();
                        c.aspect = viewport.width / viewport.height;
                        *c
                    };
                    let molecule = self.molecule.lock().unwrap();
                    let debug_infos = if debug_mode {
                        editor::collect_atom_debug_info(&molecule.atoms, camera, viewport)
                    } else {
                        Vec::new()
                    };
                    drop(molecule);

                    let mut container = div()
                        .flex_1()
                        .m_4()
                        .relative()
                        .overflow_hidden()
                        .on_mouse_down(
                            MouseButton::Left,
                            cx.listener(|this, event: &MouseDownEvent, _, cx| {
                                this.handle_click(event, cx)
                            }),
                        )
                        .on_scroll_wheel(cx.listener(|this, event: &ScrollWheelEvent, _, cx| {
                            let amount = match event.delta {
                                ScrollDelta::Pixels(delta) => f32::from(delta.y) / 80.0,
                                ScrollDelta::Lines(delta) => delta.y,
                            };
                            let mut c = this.camera.lock().unwrap();
                            c.zoom = (c.zoom * (1.0 - amount * 0.12)).clamp(0.15, 3.0);
                            cx.notify();
                        }))
                        .on_mouse_move(cx.listener(|this, event: &MouseMoveEvent, _, cx| {
                            if let Some(last) = this.last_pointer {
                                let dx = f32::from(event.position.x) - f32::from(last.x);
                                let dy = f32::from(event.position.y) - f32::from(last.y);
                                let mut c = this.camera.lock().unwrap();
                                match event.pressed_button {
                                    Some(MouseButton::Left) => {
                                        c.yaw += dx * 0.012;
                                        c.pitch = (c.pitch + dy * 0.012).clamp(-1.5, 1.5);
                                    }
                                    Some(MouseButton::Right) => {
                                        c.pan_x += dx * 0.002;
                                        c.pan_y -= dy * 0.002;
                                    }
                                    _ => {}
                                }
                            }
                            this.last_pointer = Some(event.position);
                            cx.notify();
                        }))
                        .child(MoleculeSurface::new(
                            self.surface.clone(),
                            self.viewport.clone(),
                        ));

                    if debug_mode {
                        for info in &debug_infos {
                            let rx = info.render_center.0 - viewport.x;
                            let ry = info.render_center.1 - viewport.y;
                            let px_c = info.pick_center.0 - viewport.x;
                            let py_c = info.pick_center.1 - viewport.y;

                            // A. Rendering center (Green marker) & ID label
                            container = container.child(
                                div()
                                    .absolute()
                                    .left(px(rx - 3.0))
                                    .top(px(ry - 3.0))
                                    .w(px(6.0))
                                    .h(px(6.0))
                                    .bg(rgb(0x22c55e))
                            );
                            container = container.child(
                                div()
                                    .absolute()
                                    .left(px(rx + 4.0))
                                    .top(px(ry - 8.0))
                                    .text_color(rgb(0x22c55e))
                                    .child(format!("#{}", info.atom_id))
                            );

                            // B. Picking center (Blue marker)
                            container = container.child(
                                div()
                                    .absolute()
                                    .left(px(px_c - 2.0))
                                    .top(px(py_c - 2.0))
                                    .w(px(4.0))
                                    .h(px(4.0))
                                    .bg(rgb(0x3b82f6))
                            );

                            // D. Picking boundary (4 corner markers of projection bounding box)
                            let prx = info.pick_radius_px.0;
                            let pry = info.pick_radius_px.1;
                            let corners = [
                                (px_c - prx, py_c - pry),
                                (px_c + prx, py_c - pry),
                                (px_c - prx, py_c + pry),
                                (px_c + prx, py_c + pry),
                            ];
                            for (cx_pos, cy_pos) in corners {
                                container = container.child(
                                    div()
                                        .absolute()
                                        .left(px(cx_pos - 1.0))
                                        .top(px(cy_pos - 1.0))
                                        .w(px(2.0))
                                        .h(px(2.0))
                                        .bg(rgb(0x3b82f6))
                                );
                            }
                        }

                        // C. Mouse position (Red marker)
                        if let Some(pos) = self.last_pointer {
                            let mx = f32::from(pos.x) - viewport.x;
                            let my = f32::from(pos.y) - viewport.y;
                            container = container.child(
                                div()
                                    .absolute()
                                    .left(px(mx - 4.0))
                                    .top(px(my - 4.0))
                                    .w(px(8.0))
                                    .h(px(8.0))
                                    .bg(rgb(0xef4444))
                            );
                        }
                    }

                    container
                }
            )
            .child(
                if self.debug_mode {
                    let mouse_str = self.last_pointer.map(|p| format!("({:.1}, {:.1})", f32::from(p.x), f32::from(p.y))).unwrap_or("None".into());
                    let camera = *self.camera.lock().unwrap();
                    let viewport = self.current_viewport();
                    let molecule = self.molecule.lock().unwrap();
                    let debug_infos = editor::collect_atom_debug_info(&molecule.atoms, camera, viewport);
                    drop(molecule);

                    let viewport_aspect = viewport.width / viewport.height;

                    let mut panel = div()
                        .px_4()
                        .py_2()
                        .bg(rgb(0x181b22))
                        .text_color(rgb(0xe7ebf2))
                        .flex()
                        .flex_col()
                        .gap_1()
                        .child(
                            div()
                                .flex()
                                .gap_4()
                                .items_center()
                                .child(format!("DEBUG | Mouse: {} | Viewport: {:.0}x{:.0} | Aspect: {:.3} | Sel: {:?}", mouse_str, viewport.width, viewport.height, viewport_aspect, self.selected))
                                .child(
                                    div()
                                        .px_2()
                                        .py_0()
                                        .bg(rgb(0x374151))
                                        .text_color(rgb(0xffffff))
                                        .on_mouse_down(MouseButton::Left, cx.listener(|this, _, _, cx| {
                                            this.debug_compact = !this.debug_compact;
                                            cx.notify();
                                        }))
                                        .child(format!("View: {}", if self.debug_compact { "Compact" } else { "Detailed" }))
                                )
                        );

                    if !self.debug_compact {
                        for info in debug_infos {
                            panel = panel.child(format!(
                                "ID {}({}): Render(x:{:.1}, y:{:.1}) | Pick(x:{:.1}, y:{:.1}) | Diff(dx:{:.2}, dy:{:.2})",
                                info.atom_id,
                                info.element,
                                info.render_center.0, info.render_center.1,
                                info.pick_center.0, info.pick_center.1,
                                info.diff.0, info.diff.1,
                            ));
                        }
                    } else {
                        let max_dx = debug_infos.iter().map(|i| i.diff.0.abs()).max_by(|a, b| a.total_cmp(b)).unwrap_or(0.0);
                        let max_dy = debug_infos.iter().map(|i| i.diff.1.abs()).max_by(|a, b| a.total_cmp(b)).unwrap_or(0.0);
                        panel = panel.child(format!("Atoms: {} | Max Diff: dx={:.2}px, dy={:.2}px (Green: Render, Blue: Pick, Red: Mouse)", debug_infos.len(), max_dx, max_dy));
                    }
                    panel
                } else {
                    div()
                }
            )
    }
}

fn render_loop(
    surface: WgpuSurfaceHandle,
    molecule: Arc<Mutex<Molecule>>,
    camera: Arc<Mutex<Camera>>,
    selected: Arc<Mutex<Vec<u64>>>,
) {
    while surface.back_buffer_view().is_none() {
        thread::sleep(Duration::from_millis(10));
    }
    let device = surface.device();
    let queue = surface.queue();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("molecule"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
            "../shade/molecule.wgsl"
        ))),
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
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });
    let mut depth_texture: Option<(u32, u32, wgpu::Texture, wgpu::TextureView)> = None;
    loop {
        surface.wait_for_present();
        let Some((view, (width, height))) = surface.back_view_with_size() else {
            continue;
        };
        if width == 0 || height == 0 {
            continue;
        }
        let depth_view = match &depth_texture {
            Some((w, h, _, view)) if *w == width && *h == height => view.clone(),
            _ => {
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("depth texture"),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Depth32Float,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                });
                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                let v = view.clone();
                depth_texture = Some((width, height, texture, view));
                v
            }
        };
        let c = *camera.lock().unwrap();
        let aspect = if c.aspect > 0.0 { c.aspect } else { width as f32 / height as f32 };
        let sel = selected.lock().unwrap();
        let verts = mesh(&molecule.lock().unwrap(), &sel);
        drop(sel);
        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("molecule vertices"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        queue.write_buffer(
            &ub,
            0,
            bytemuck::cast_slice(&[c.yaw, c.pitch, c.zoom, aspect, c.pan_x, c.pan_y, 0.0, 0.0]),
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
                            r: 0.03,
                            g: 0.05,
                            b: 0.07,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
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
    let mut molecule: Molecule = serde_json::from_str(&text).expect("valid molecule JSON");
    normalize_ids(&mut molecule);
    Application::new().run(move |cx| {
        cx.open_window(WindowOptions::default(), move |w, cx| {
            w.set_window_title("qm_portal");
            w.resize(size(px(900.0), px(600.0)));
            let surface = w
                .create_wgpu_surface(900, 600, wgpu::TextureFormat::Rgba8UnormSrgb)
                .expect("WgpuSurface unavailable");
            let draw_surface = surface.clone();
            let molecule = Arc::new(Mutex::new(molecule));
            let draw_molecule = molecule.clone();
            let camera = Arc::new(Mutex::new(Camera::default()));
            let draw_camera = camera.clone();
            let selected = Arc::new(Mutex::new(Vec::new()));
            let draw_selected = selected.clone();
            thread::spawn(move || render_loop(draw_surface, draw_molecule, draw_camera, draw_selected));
            cx.new(|_| View {
                surface,
                molecule,
                editor: editor::EditorState::new(),
                selected: Vec::new(),
                selected_arc: selected,
                camera,
                last_pointer: None,
                edit_mode: EditMode::Select,
                numeric_buffer: String::new(),
                slider_last_x: None,
                slider_value: None,
                fragment_mode: true,
                debug_mode: false,
                debug_compact: true,
                viewport: Arc::new(Mutex::new(Viewport {
                    x: 16.0,
                    y: 132.0,
                    width: 868.0,
                    height: 452.0,
                })),
            })
        })
        .unwrap();
    });
}
