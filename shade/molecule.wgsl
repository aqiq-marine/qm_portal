struct U {
    yaw: f32,
    pitch: f32,
    zoom: f32,
    aspect: f32,
    pan_x: f32,
    pan_y: f32,
    _pad2: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> u: U;

struct I {
    @location(0) p: vec3<f32>,
    @location(1) c: vec3<f32>,
};

struct O {
    @builtin(position) p: vec4<f32>,
    @location(0) c: vec3<f32>,
};

@vertex
fn vs(i: I) -> O {
    let cy = cos(u.yaw);
    let sy = sin(u.yaw);
    let x = i.p.x * cy - i.p.z * sy;
    let z = i.p.x * sy + i.p.z * cy;
    let cp = cos(u.pitch);
    let sp = sin(u.pitch);
    let y = i.p.y * cp - z * sp;
    let rz = i.p.y * sp + z * cp;
    var o: O;
    o.p = vec4<f32>(
        (x + u.pan_x) * u.zoom / u.aspect,
        (y + u.pan_y) * u.zoom,
        rz * 0.12 + 0.5,
        1.0,
    );
    o.c = i.c;
    return o;
}

@fragment
fn fs(i: O) -> @location(0) vec4<f32> {
    return vec4<f32>(i.c, 1);
}
