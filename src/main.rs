#![recursion_limit = "256"]
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

fn atom_radius(atom: &Atom) -> f32 {
    if atom.element == "H" { 0.16 } else { 0.25 }
}

// Apply the same yaw-then-pitch rotation as shade/molecule.wgsl.
// Translation, zoom, and aspect are deliberately handled separately.
fn atom_view_position(atom: &Atom, camera: Camera) -> [f32; 3] {
    let cy = camera.yaw.cos();
    let sy = camera.yaw.sin();
    let x = atom.x * cy - atom.z * sy;
    let z = atom.x * sy + atom.z * cy;

    let cp = camera.pitch.cos();
    let sp = camera.pitch.sin();
    let y = atom.y * cp - z * sp;
    let view_z = atom.y * sp + z * cp;

    [x, y, view_z]
}

// CPU equivalent of the WGSL vertex shader's x/y projection.
// Returned coordinates are in the same window-pixel coordinate system
// consumed by MouseDownEvent::position.
fn project_atom_to_screen(atom: &Atom, camera: Camera, viewport: Viewport) -> Option<(f32, f32)> {
    if viewport.width <= 0.0 || viewport.height <= 0.0 || camera.zoom <= 0.0 {
        return None;
    }

    let [x, y, _] = atom_view_position(atom, camera);
    let aspect = viewport.width / viewport.height;
    let ndc_x = (x + camera.pan_x) * camera.zoom / aspect;
    let ndc_y = (y + camera.pan_y) * camera.zoom;

    Some((
        viewport.x + (ndc_x + 1.0) * viewport.width * 0.5,
        viewport.y + (1.0 - ndc_y) * viewport.height * 0.5,
    ))
}

// Orthographic ray/sphere picking. The shader maps smaller view-space Z
// to smaller depth values and uses CompareFunction::Less, so the camera is
// on the -Z side: rays travel along +Z and the smallest intersection t wins.
fn pick_atom_at(
    atoms: &[Atom],
    camera: Camera,
    viewport: Viewport,
    screen_x: f32,
    screen_y: f32,
) -> Option<u64> {
    if viewport.width <= 0.0
        || viewport.height <= 0.0
        || camera.zoom <= 0.0
        || !screen_x.is_finite()
        || !screen_y.is_finite()
    {
        return None;
    }

    let local_x = screen_x - viewport.x;
    let local_y = screen_y - viewport.y;
    if local_x < 0.0
        || local_y < 0.0
        || local_x > viewport.width
        || local_y > viewport.height
    {
        return None;
    }

    let ndc_x = 2.0 * local_x / viewport.width - 1.0;
    let ndc_y = 1.0 - 2.0 * local_y / viewport.height;
    let aspect = viewport.width / viewport.height;
    let ray_x = ndc_x * aspect / camera.zoom - camera.pan_x;
    let ray_y = ndc_y / camera.zoom - camera.pan_y;

    // Start before the closest (smallest-Z) sphere and travel toward +Z.
    let min_view_z = atoms
        .iter()
        .map(|atom| atom_view_position(atom, camera)[2] - atom_radius(atom))
        .fold(f32::INFINITY, f32::min);
    if !min_view_z.is_finite() {
        return None;
    }
    let ray_origin_z = min_view_z - 1.0;

    atoms
        .iter()
        .filter_map(|atom| {
            let center = atom_view_position(atom, camera);
            let radius = atom_radius(atom);
            let dx = ray_x - center[0];
            let dy = ray_y - center[1];
            let discriminant = radius * radius - dx * dx - dy * dy;
            if discriminant < 0.0 {
                return None;
            }

            let intersection_z = center[2] - discriminant.sqrt();
            let t = intersection_z - ray_origin_z;
            (t >= 0.0).then_some((t, atom.id))
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, id)| id)
}

#[cfg(test)]
mod picking_tests {
    use super::*;

    fn atom(id: u64, element: &str, x: f32, y: f32, z: f32) -> Atom {
        Atom { id, element: element.into(), x, y, z }
    }

    fn viewport() -> Viewport {
        Viewport { x: 13.0, y: 29.0, width: 868.0, height: 468.0 }
    }

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1e-3,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn projected_atom_center_can_be_picked() {
        let a = atom(7, "C", 1.2, -0.7, 2.1);
        let camera = Camera::default();
        let viewport = viewport();
        let (x, y) = project_atom_to_screen(&a, camera, viewport).unwrap();

        assert_eq!(pick_atom_at(&[a], camera, viewport, x, y), Some(7));
    }

    #[test]
    fn picking_round_trips_with_rotation_zoom_and_pan() {
        let a = atom(42, "C", -1.3, 0.8, 2.4);
        let camera = Camera {
            yaw: 0.73,
            pitch: -0.41,
            zoom: 1.7,
            pan_x: 0.23,
            pan_y: -0.17,
        };
        let viewport = viewport();
        let (x, y) = project_atom_to_screen(&a, camera, viewport).unwrap();

        assert_eq!(pick_atom_at(&[a], camera, viewport, x, y), Some(42));
    }

    #[test]
    fn sphere_edge_is_pickable_but_point_outside_is_not() {
        let a = atom(3, "C", 0.0, 0.0, 0.0);
        let camera = Camera::default();
        let viewport = viewport();
        let (center_x, center_y) = project_atom_to_screen(&a, camera, viewport).unwrap();

        // The projected horizontal radius is radius * zoom / aspect.
        let pixel_radius = atom_radius(&a) * camera.zoom
            / (viewport.width / viewport.height) * viewport.width * 0.5;
        assert_eq!(
            pick_atom_at(&[a.clone()], camera, viewport, center_x + pixel_radius * 0.99, center_y),
            Some(3)
        );
        assert_eq!(
            pick_atom_at(&[a], camera, viewport, center_x + pixel_radius * 1.01, center_y),
            None
        );
    }

    #[test]
    fn nearest_visible_sphere_wins() {
        // Smaller view-space Z is closer according to the WGSL depth mapping.
        let near = atom(1, "C", 0.0, 0.0, -1.0);
        let far = atom(2, "C", 0.0, 0.0, 1.0);
        let camera = Camera::default();
        let viewport = viewport();
        let (x, y) = project_atom_to_screen(&near, camera, viewport).unwrap();

        assert_eq!(pick_atom_at(&[far, near], camera, viewport, x, y), Some(1));
    }

    #[test]
    fn clicks_outside_viewport_do_not_pick() {
        let a = atom(1, "C", 0.0, 0.0, 0.0);
        assert_eq!(
            pick_atom_at(&[a], Camera::default(), viewport(), 0.0, 0.0),
            None
        );
    }

    #[test]
    fn hydrogen_uses_its_rendered_radius() {
        let h = atom(9, "H", 0.0, 0.0, 0.0);
        let camera = Camera::default();
        let viewport = viewport();
        let (x, y) = project_atom_to_screen(&h, camera, viewport).unwrap();

        let c = atom(8, "C", 0.0, 0.0, 0.0);
        let c_radius_px = atom_radius(&c) * camera.zoom
            / (viewport.width / viewport.height) * viewport.width * 0.5;
        assert_eq!(
            pick_atom_at(&[h], camera, viewport, x + c_radius_px * 0.9, y),
            None
        );
    }
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

#[derive(Clone, Copy, Debug)]
struct Viewport {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
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
    fragment_mode: bool,
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
        let window_width = 900.0_f32;
        let window_height = 600.0_f32;
        let margin = 16.0_f32;
        let summary_height = 44.0;
        let toolbar_height = 40.0;
        let has_edit_panel = matches!(self.edit_mode, EditMode::BondLength | EditMode::BondAngle | EditMode::Dihedral)
            && match self.edit_mode {
                EditMode::BondLength => self.selected.len() == 2,
                EditMode::BondAngle => self.selected.len() == 3,
                EditMode::Dihedral => self.selected.len() == 4,
                _ => false,
            };
        let edit_panel_height = if has_edit_panel { 40.0 } else { 0.0 };
        
        let top_offset = margin + summary_height + margin + toolbar_height + edit_panel_height;
        let x = margin;
        let y = top_offset + margin;
        let width = window_width - (margin * 2.0);
        let height = window_height - y - margin;

        Viewport { x, y, width, height }
    }

    fn pick_atom(&self, position: Point<Pixels>) -> Option<u64> {
        let viewport = self.current_viewport();
        let camera = *self.camera.lock().unwrap();
        let molecule = self.molecule.lock().unwrap();

        pick_atom_at(
            &molecule.atoms,
            camera,
            viewport,
            f32::from(position.x),
            f32::from(position.y),
        )
    }

    fn update_selection(&mut self, new_selected: Vec<u64>) {
        self.selected = new_selected;
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
                BondType::Single => BondType::Double,
                BondType::Double => BondType::Triple,
                BondType::Triple => BondType::Aromatic,
                BondType::Aromatic => BondType::Single,
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
        match mode {
            EditMode::BondLength if self.selected.len() == 2 => self.edit_mode = mode,
            EditMode::BondAngle if self.selected.len() == 3 => self.edit_mode = mode,
            EditMode::Dihedral if self.selected.len() == 4 => self.edit_mode = mode,
            EditMode::Select => self.edit_mode = mode,
            _ => {}
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

    fn adjust_value(&mut self, delta: f64) {
        let molecule = self.molecule.lock().unwrap();
        let current = match self.edit_mode {
            EditMode::BondLength if self.selected.len() == 2 => {
                editor::distance(&molecule, self.selected[0], self.selected[1]).unwrap_or(1.0)
            }
            EditMode::BondAngle if self.selected.len() == 3 => {
                editor::angle(&molecule, self.selected[0], self.selected[1], self.selected[2]).unwrap_or(1.0).to_degrees()
            }
            EditMode::Dihedral if self.selected.len() == 4 => {
                editor::dihedral(&molecule, self.selected[0], self.selected[1], self.selected[2], self.selected[3]).unwrap_or(0.0).to_degrees()
            }
            _ => return,
        };
        drop(molecule);
        let new_val = current + delta;
        self.apply_value(new_val);
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
                    )
                    .child(match self.edit_mode {
                        EditMode::BondLength if self.selected.len() == 2 => {
                            div()
                                .flex()
                                .gap_4()
                                .items_center()
                                .child(format!("Bond Length: {}", current_val_str))
                                .child(
                                    div()
                                        .px_2()
                                        .py_1()
                                        .bg(rgb(0x374151))
                                        .on_mouse_down(MouseButton::Left, cx.listener(|this, _, _, cx| {
                                            this.adjust_value(-0.1);
                                            cx.notify();
                                        }))
                                        .child("-0.1")
                                )
                                .child(
                                    div()
                                        .px_2()
                                        .py_1()
                                        .bg(rgb(0x374151))
                                        .on_mouse_down(MouseButton::Left, cx.listener(|this, _, _, cx| {
                                            this.adjust_value(0.1);
                                            cx.notify();
                                        }))
                                        .child("+0.1")
                                )
                        }
                        EditMode::BondAngle if self.selected.len() == 3 => {
                            div()
                                .flex()
                                .gap_4()
                                .items_center()
                                .child(format!("Bond Angle: {}", current_val_str))
                                .child(
                                    div()
                                        .px_2()
                                        .py_1()
                                        .bg(rgb(0x374151))
                                        .on_mouse_down(MouseButton::Left, cx.listener(|this, _, _, cx| {
                                            this.adjust_value(-5.0);
                                            cx.notify();
                                        }))
                                        .child("-5°")
                                )
                                .child(
                                    div()
                                        .px_2()
                                        .py_1()
                                        .bg(rgb(0x374151))
                                        .on_mouse_down(MouseButton::Left, cx.listener(|this, _, _, cx| {
                                            this.adjust_value(5.0);
                                            cx.notify();
                                        }))
                                        .child("+5°")
                                )
                        }
                        EditMode::Dihedral if self.selected.len() == 4 => {
                            div()
                                .flex()
                                .gap_4()
                                .items_center()
                                .child(format!("Dihedral Angle: {}", current_val_str))
                                .child(
                                    div()
                                        .px_2()
                                        .py_1()
                                        .bg(rgb(0x374151))
                                        .on_mouse_down(MouseButton::Left, cx.listener(|this, _, _, cx| {
                                            this.adjust_value(-5.0);
                                            cx.notify();
                                        }))
                                        .child("-5°")
                                )
                                .child(
                                    div()
                                        .px_2()
                                        .py_1()
                                        .bg(rgb(0x374151))
                                        .on_mouse_down(MouseButton::Left, cx.listener(|this, _, _, cx| {
                                            this.adjust_value(5.0);
                                            cx.notify();
                                        }))
                                        .child("+5°")
                                )
                        }
                        _ => div().child(format!("Mode: {} | {}", self.edit_mode.label(), current_val_str)),
                    })
            )
            .child(
                div()
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
                    .child(wgpu_surface(self.surface.clone()).absolute().inset_0()),
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
        let aspect = width as f32 / height as f32;
        let c = *camera.lock().unwrap();
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
                fragment_mode: false,
            })
        })
        .unwrap();
    });
}
