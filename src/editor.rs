use std::collections::{HashMap, HashSet};

use super::{BondType, Molecule};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
    pub fn cross(self, other: Self) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }
    pub fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }
    pub fn normalized(self) -> Result<Self, EditError> {
        let norm = self.norm();
        if norm < 1.0e-12 {
            return Err(EditError::DegenerateGeometry);
        }
        Ok(self / norm)
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}
impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}
impl std::ops::Mul<f64> for Vec3 {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}
impl std::ops::Div<f64> for Vec3 {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Self::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

fn get(molecule: &Molecule, id: u64) -> Result<(usize, Vec3), EditError> {
    molecule
        .atoms
        .iter()
        .enumerate()
        .find(|(_, atom)| atom.id == id)
        .map(|(index, atom)| {
            (
                index,
                Vec3::new(atom.x as f64, atom.y as f64, atom.z as f64),
            )
        })
        .ok_or(EditError::UnknownAtom(id))
}

fn set(molecule: &mut Molecule, index: usize, p: Vec3) {
    molecule.atoms[index].x = p.x as f32;
    molecule.atoms[index].y = p.y as f32;
    molecule.atoms[index].z = p.z as f32;
}

pub(crate) fn distance(molecule: &Molecule, a: u64, b: u64) -> Result<f64, EditError> {
    let (_, pa) = get(molecule, a)?;
    let (_, pb) = get(molecule, b)?;
    Ok((pb - pa).norm())
}

pub(crate) fn angle(molecule: &Molecule, a: u64, center: u64, c: u64) -> Result<f64, EditError> {
    let (_, pa) = get(molecule, a)?;
    let (_, pb) = get(molecule, center)?;
    let (_, pc) = get(molecule, c)?;
    let u = (pa - pb).normalized()?;
    let v = (pc - pb).normalized()?;
    Ok(u.dot(v).clamp(-1.0, 1.0).acos())
}

pub(crate) fn dihedral(
    molecule: &Molecule,
    a: u64,
    b: u64,
    c: u64,
    d: u64,
) -> Result<f64, EditError> {
    let (_, pa) = get(molecule, a)?;
    let (_, pb) = get(molecule, b)?;
    let (_, pc) = get(molecule, c)?;
    let (_, pd) = get(molecule, d)?;
    let b1 = pb - pa;
    let b2 = pc - pb;
    let b3 = pd - pc;
    let n1 = b1.cross(b2).normalized()?;
    let n2 = b2.cross(b3).normalized()?;
    let axis = b2.normalized()?;
    Ok(n1.cross(n2).dot(axis).atan2(n1.dot(n2)))
}

pub(crate) fn connected_fragment(
    molecule: &Molecule,
    root: u64,
    excluded: Option<(u64, u64)>,
) -> Result<HashSet<u64>, EditError> {
    get(molecule, root)?;
    let mut graph: HashMap<u64, Vec<u64>> = HashMap::new();
    for bond in &molecule.bonds {
        let a = molecule
            .atoms
            .get(bond.from)
            .ok_or(EditError::BrokenBond(bond.id))?
            .id;
        let b = molecule
            .atoms
            .get(bond.to)
            .ok_or(EditError::BrokenBond(bond.id))?
            .id;
        if excluded.is_some_and(|(x, y)| (a == x && b == y) || (a == y && b == x)) {
            continue;
        }
        graph.entry(a).or_default().push(b);
        graph.entry(b).or_default().push(a);
    }
    let mut seen = HashSet::from([root]);
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        for next in graph.get(&id).into_iter().flatten() {
            if seen.insert(*next) {
                stack.push(*next);
            }
        }
    }
    Ok(seen)
}

#[derive(Debug, PartialEq)]
pub(crate) enum EditError {
    UnknownAtom(u64),
    UnknownBond(u64),
    BrokenBond(u64),
    InvalidValue,
    DegenerateGeometry,
    RingConstraint,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum MoveSelection {
    OneAtom,
    ConnectedFragment,
}

#[derive(Clone, Debug)]
pub(crate) enum EditCommand {
    SetBondOrder {
        bond_id: u64,
        order: BondType,
    },
    SetBondLength {
        from: u64,
        to: u64,
        length: f64,
        move_selection: MoveSelection,
    },
    SetBondAngle {
        a: u64,
        center: u64,
        c: u64,
        angle: f64,
        move_selection: MoveSelection,
    },
    SetDihedral {
        a: u64,
        b: u64,
        c: u64,
        d: u64,
        angle: f64,
    },
}

#[derive(Clone)]
struct Snapshot {
    atoms: Vec<(usize, (f32, f32, f32))>,
    bonds: Vec<(usize, BondType)>,
}

pub(crate) struct EditorState {
    history: Vec<(Snapshot, Snapshot)>,
    redo: Vec<(Snapshot, Snapshot)>,
}
impl EditorState {
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            redo: Vec::new(),
        }
    }
    pub fn apply(
        &mut self,
        molecule: &mut Molecule,
        command: EditCommand,
    ) -> Result<(), EditError> {
        let before = snapshot(molecule);
        apply(molecule, command)?;
        let after = snapshot(molecule);
        self.history.push((before, after));
        self.redo.clear();
        Ok(())
    }
    pub fn undo(&mut self, molecule: &mut Molecule) -> bool {
        if let Some((before, after)) = self.history.pop() {
            restore(molecule, &before);
            self.redo.push((before, after));
            true
        } else {
            false
        }
    }
    pub fn redo(&mut self, molecule: &mut Molecule) -> bool {
        if let Some((before, after)) = self.redo.pop() {
            restore(molecule, &after);
            self.history.push((before, after));
            true
        } else {
            false
        }
    }
}

fn snapshot(molecule: &Molecule) -> Snapshot {
    Snapshot {
        atoms: molecule
            .atoms
            .iter()
            .enumerate()
            .map(|(i, a)| (i, (a.x, a.y, a.z)))
            .collect(),
        bonds: molecule
            .bonds
            .iter()
            .enumerate()
            .map(|(i, bond)| (i, bond.order.clone()))
            .collect(),
    }
}

fn restore(molecule: &mut Molecule, snapshot: &Snapshot) {
    for (index, (x, y, z)) in &snapshot.atoms {
        molecule.atoms[*index].x = *x;
        molecule.atoms[*index].y = *y;
        molecule.atoms[*index].z = *z;
    }
    for (index, order) in &snapshot.bonds {
        molecule.bonds[*index].order = order.clone();
    }
}

fn rotate(p: Vec3, origin: Vec3, axis: Vec3, angle: f64) -> Vec3 {
    let v = p - origin;
    let c = angle.cos();
    let s = angle.sin();
    origin + v * c + axis.cross(v) * s + axis * (axis.dot(v) * (1.0 - c))
}

fn apply(molecule: &mut Molecule, command: EditCommand) -> Result<(), EditError> {
    match command {
        EditCommand::SetBondOrder { bond_id, order } => {
            let bond = molecule
                .bonds
                .iter_mut()
                .find(|b| b.id == bond_id)
                .ok_or(EditError::UnknownBond(bond_id))?;
            bond.order = order;
        }
        EditCommand::SetBondLength {
            from,
            to,
            length,
            move_selection,
        } => {
            if length <= 0.0 {
                return Err(EditError::InvalidValue);
            }
            let (_, pa) = get(molecule, from)?;
            let (to_index, pb) = get(molecule, to)?;
            let direction = (pb - pa).normalized()?;
            let delta = direction * (length - (pb - pa).norm());
            let ids = match move_selection {
                MoveSelection::OneAtom => HashSet::from([to]),
                MoveSelection::ConnectedFragment => {
                    connected_fragment(molecule, to, Some((from, to)))?
                }
            };
            for id in ids {
                let (index, p) = get(molecule, id)?;
                set(molecule, index, p + delta);
            }
            let _ = to_index;
        }
        EditCommand::SetBondAngle {
            a,
            center,
            c,
            angle: target,
            move_selection,
        } => {
            if !(0.0 < target && target < std::f64::consts::PI) {
                return Err(EditError::InvalidValue);
            }
            let current = angle(molecule, a, center, c)?;
            let (_, pc) = get(molecule, c)?;
            let (_, pb) = get(molecule, center)?;
            let axis = (get(molecule, a)?.1 - pb).cross(pc - pb).normalized()?;
            let ids = match move_selection {
                MoveSelection::OneAtom => HashSet::from([c]),
                MoveSelection::ConnectedFragment => {
                    connected_fragment(molecule, c, Some((center, c)))?
                }
            };
            for id in ids {
                let (index, p) = get(molecule, id)?;
                set(molecule, index, rotate(p, pb, axis, target - current));
            }
        }
        EditCommand::SetDihedral {
            a,
            b,
            c,
            d,
            angle: target,
        } => {
            let current = dihedral(molecule, a, b, c, d)?;
            let (_, pb) = get(molecule, b)?;
            let (_, pc) = get(molecule, c)?;
            let axis = (pc - pb).normalized()?;
            let ids = connected_fragment(molecule, d, Some((b, c)))?;
            if ids.contains(&b) {
                return Err(EditError::RingConstraint);
            }
            for id in ids {
                let (index, p) = get(molecule, id)?;
                set(molecule, index, rotate(p, pb, axis, target - current));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Atom, Bond};
    fn water() -> Molecule {
        Molecule {
            charge: 0,
            spin: 1,
            atoms: vec![
                Atom {
                    id: 1,
                    element: "O".into(),
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                Atom {
                    id: 2,
                    element: "H".into(),
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                },
                Atom {
                    id: 3,
                    element: "H".into(),
                    x: 0.0,
                    y: 1.0,
                    z: 0.0,
                },
            ],
            bonds: vec![
                Bond {
                    id: 1,
                    from: 0,
                    to: 1,
                    order: BondType::Single,
                },
                Bond {
                    id: 2,
                    from: 0,
                    to: 2,
                    order: BondType::Single,
                },
            ],
        }
    }
    #[test]
    fn geometry_is_computed() {
        let m = water();
        assert!((distance(&m, 1, 2).unwrap() - 1.0).abs() < 1e-10);
        assert!((angle(&m, 2, 1, 3).unwrap() - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    }
    #[test]
    fn fragment_excludes_bond() {
        let m = water();
        assert_eq!(
            connected_fragment(&m, 2, Some((1, 2))).unwrap(),
            HashSet::from([2])
        );
    }
    #[test]
    fn bond_length_and_undo() {
        let mut m = water();
        let mut e = EditorState::new();
        e.apply(
            &mut m,
            EditCommand::SetBondLength {
                from: 1,
                to: 2,
                length: 2.0,
                move_selection: MoveSelection::OneAtom,
            },
        )
        .unwrap();
        assert!((distance(&m, 1, 2).unwrap() - 2.0).abs() < 1e-6);
        assert!(e.undo(&mut m));
        assert!((distance(&m, 1, 2).unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn bond_angle_edit_reaches_target() {
        let mut m = water();
        let mut e = EditorState::new();
        e.apply(
            &mut m,
            EditCommand::SetBondAngle {
                a: 2,
                center: 1,
                c: 3,
                angle: std::f64::consts::FRAC_PI_4,
                move_selection: MoveSelection::OneAtom,
            },
        )
        .unwrap();
        assert!((angle(&m, 2, 1, 3).unwrap() - std::f64::consts::FRAC_PI_4).abs() < 1e-6);
    }

    #[test]
    fn dihedral_edit_reaches_target() {
        let mut m = Molecule {
            charge: 0,
            spin: 1,
            atoms: vec![
                Atom {
                    id: 1,
                    element: "C".into(),
                    x: 0.0,
                    y: 1.0,
                    z: 0.0,
                },
                Atom {
                    id: 2,
                    element: "C".into(),
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                Atom {
                    id: 3,
                    element: "C".into(),
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                },
                Atom {
                    id: 4,
                    element: "C".into(),
                    x: 1.0,
                    y: 1.0,
                    z: 0.0,
                },
            ],
            bonds: vec![
                Bond {
                    id: 1,
                    from: 0,
                    to: 1,
                    order: BondType::Single,
                },
                Bond {
                    id: 2,
                    from: 1,
                    to: 2,
                    order: BondType::Single,
                },
                Bond {
                    id: 3,
                    from: 2,
                    to: 3,
                    order: BondType::Single,
                },
            ],
        };
        let mut e = EditorState::new();
        e.apply(
            &mut m,
            EditCommand::SetDihedral {
                a: 1,
                b: 2,
                c: 3,
                d: 4,
                angle: std::f64::consts::FRAC_PI_2,
            },
        )
        .unwrap();
        assert!((dihedral(&m, 1, 2, 3, 4).unwrap() - std::f64::consts::FRAC_PI_2).abs() < 1e-6);
    }

    #[test]
    fn bond_order_edit_is_undoable() {
        let mut m = water();
        let mut e = EditorState::new();
        e.apply(
            &mut m,
            EditCommand::SetBondOrder {
                bond_id: 1,
                order: BondType::Double,
            },
        )
        .unwrap();
        assert_eq!(m.bonds[0].order, BondType::Double);
        assert!(e.undo(&mut m));
        assert_eq!(m.bonds[0].order, BondType::Single);
    }

    fn atom(id: u64, element: &str, x: f32, y: f32, z: f32) -> Atom {
        Atom { id, element: element.into(), x, y, z }
    }

    fn viewport() -> PickViewport {
        PickViewport { x: 13.0, y: 29.0, width: 868.0, height: 468.0 }
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
        let a = atom(42, "C", -0.3, 0.2, 0.4);
        let camera = Camera {
            yaw: 0.73,
            pitch: -0.41,
            zoom: 1.7,
            ..Camera::default()
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

#[derive(Clone, Copy)]
pub(crate) struct Camera {
    pub yaw: f32,
    pub pitch: f32,
    pub zoom: f32,
    pub pan_x: f32,
    pub pan_y: f32,
    pub aspect: f32,
}
impl Default for Camera {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.0,
            zoom: 0.55,
            pan_x: 0.0,
            pan_y: 0.0,
            aspect: 868.0 / 468.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PickViewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

fn atom_radius(atom: &super::Atom) -> f32 {
    if atom.element == "H" { 0.16 } else { 0.25 }
}

fn atom_view_position(atom: &super::Atom, camera: Camera) -> [f32; 3] {
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

pub(crate) fn project_atom_to_screen(atom: &super::Atom, camera: Camera, viewport: PickViewport) -> Option<(f32, f32)> {
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

pub(crate) fn pick_atom_at(
    atoms: &[super::Atom],
    camera: Camera,
    viewport: PickViewport,
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

#[derive(Clone, Debug)]
pub(crate) struct AtomDebugInfo {
    pub atom_id: u64,
    pub element: String,
    pub render_center: (f32, f32),
    pub pick_center: (f32, f32),
    pub diff: (f32, f32),
    pub render_radius_px: (f32, f32),
    pub pick_radius_px: (f32, f32),
}

pub(crate) fn collect_atom_debug_info(
    atoms: &[super::Atom],
    camera: Camera,
    viewport: PickViewport,
) -> Vec<AtomDebugInfo> {
    let mut infos = Vec::new();
    if viewport.width <= 0.0 || viewport.height <= 0.0 || camera.zoom <= 0.0 {
        return infos;
    }
    let aspect = viewport.width / viewport.height;

    for atom in atoms {
        let Some(render_center) = project_atom_to_screen(atom, camera, viewport) else {
            continue;
        };
        let center = atom_view_position(atom, camera);
        let ndc_x = (center[0] + camera.pan_x) * camera.zoom / aspect;
        let ndc_y = (center[1] + camera.pan_y) * camera.zoom;
        let pick_center = (
            viewport.x + (ndc_x + 1.0) * viewport.width * 0.5,
            viewport.y + (1.0 - ndc_y) * viewport.height * 0.5,
        );

        let diff = (render_center.0 - pick_center.0, render_center.1 - pick_center.1);
        let r = atom_radius(atom);

        let render_radius_px = (
            r * camera.zoom * viewport.width * 0.5 / aspect,
            r * camera.zoom * viewport.height * 0.5,
        );
        let pick_radius_px = (
            r * camera.zoom * viewport.width * 0.5 / aspect,
            r * camera.zoom * viewport.height * 0.5,
        );

        infos.push(AtomDebugInfo {
            atom_id: atom.id,
            element: atom.element.clone(),
            render_center,
            pick_center,
            diff,
            render_radius_px,
            pick_radius_px,
        });
    }
    infos
}

