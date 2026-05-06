"""Generate 3 STL turbine models with different blade counts and profiles."""
import numpy as np
import struct
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), 'models')

def write_stl(filename, tris):
    with open(filename, 'wb') as f:
        f.write(b' ' * 80)
        f.write(struct.pack('<I', len(tris)))
        for n, a, b, c in tris:
            f.write(struct.pack('<3f', *n))
            f.write(struct.pack('<3f', *a))
            f.write(struct.pack('<3f', *b))
            f.write(struct.pack('<3f', *c))
            f.write(struct.pack('<H', 0))

def norm(v):
    v = np.asarray(v, float)
    d = np.linalg.norm(v)
    return v / d if d > 1e-12 else np.array([0., 0., 1.])

def fnorm(a, b, c):
    a, b, c = map(np.asarray, (a, b, c))
    return tuple(norm(np.cross(b - a, c - a)))

def quad(tris, a, b, c, d):
    """Add CCW quad as two triangles with auto normals."""
    a, b, c, d = map(np.asarray, (a, b, c, d))
    tris.append((fnorm(a, b, c), tuple(a), tuple(b), tuple(c)))
    tris.append((fnorm(a, c, d), tuple(a), tuple(c), tuple(d)))

def cylinder(tris, r, h, segs=32):
    """Solid cylinder centred at origin, axis=Z."""
    for i in range(segs):
        a0, a1 = 2*np.pi*i/segs, 2*np.pi*(i+1)/segs
        x0, y0 = r*np.cos(a0), r*np.sin(a0)
        x1, y1 = r*np.cos(a1), r*np.sin(a1)
        quad(tris, [x0,y0,-h/2], [x1,y1,-h/2], [x1,y1,h/2], [x0,y0,h/2])
        tris.append(((0,0, 1), (0,0, h/2), (x0,y0, h/2), (x1,y1, h/2)))
        tris.append(((0,0,-1), (0,0,-h/2), (x1,y1,-h/2), (x0,y0,-h/2)))

def rot_z(p, a):
    """Rotate point [x,y,z] around Z by angle a."""
    c, s = np.cos(a), np.sin(a)
    x, y, z = p
    return np.array([c*x - s*y, s*x + c*y, z])

def section_pts(r, chord, twist, thick):
    """
    4 corners of a blade section at radius r.
    Chord along local-Y, thickness along Z, rotated by twist in YZ plane.
    Returns [le_top, le_bot, te_top, te_bot] in blade local frame [r, chord, axial].
    """
    hw, ht = chord/2, thick/2
    ct, st = np.cos(twist), np.sin(twist)
    def ry(y, z): return ct*y - st*z, st*y + ct*z
    le_y_t, le_z_t = ry(-hw,  ht)
    le_y_b, le_z_b = ry(-hw, -ht)
    te_y_t, te_z_t = ry( hw,  ht)
    te_y_b, te_z_b = ry( hw, -ht)
    return [
        np.array([r, le_y_t, le_z_t]),  # 0 LE-top
        np.array([r, le_y_b, le_z_b]),  # 1 LE-bot
        np.array([r, te_y_t, te_z_t]),  # 2 TE-top
        np.array([r, te_y_b, te_z_b]),  # 3 TE-bot
    ]

def blade(tris, n_blades, blade_idx,
          r0, r1, n_sec,
          chord_root, chord_tip,
          twist_root, twist_tip,
          thick, sweep_z):
    """Build one blade and add its triangles to tris."""
    angle = 2*np.pi * blade_idx / n_blades
    sections = []
    for j in range(n_sec + 1):
        t = j / n_sec
        r = r0 + t * (r1 - r0)
        chord = chord_root + t * (chord_tip - chord_root)
        twist = twist_root + t * (twist_tip - twist_root)
        z_sw = t * sweep_z
        pts = section_pts(r, chord, twist, thick)
        # apply sweep and then rotate around Z into world frame
        pts_world = [rot_z([p[0], p[1], p[2] + z_sw], angle) for p in pts]
        sections.append(pts_world)

    for i in range(n_sec):
        p = sections[i];   q = sections[i+1]
        # suction (top) face: outward = away from pressure side
        quad(tris, p[0], p[2], q[2], q[0])
        # pressure (bottom) face: reversed winding
        quad(tris, p[1], q[1], q[3], p[3])
        # leading edge strip
        quad(tris, p[1], p[0], q[0], q[1])
        # trailing edge strip
        quad(tris, p[2], p[3], q[3], q[2])

    # tip cap
    tip = sections[-1]
    tris.append((fnorm(tip[0], tip[2], tip[1]), tuple(tip[0]), tuple(tip[2]), tuple(tip[3])))
    tris.append((fnorm(tip[0], tip[3], tip[1]), tuple(tip[0]), tuple(tip[3]), tuple(tip[1])))
    # root cap
    root = sections[0]
    tris.append((fnorm(root[0], root[1], root[2]), tuple(root[0]), tuple(root[1]), tuple(root[3])))
    tris.append((fnorm(root[0], root[3], root[2]), tuple(root[0]), tuple(root[3]), tuple(root[2])))


def make_turbine(num_blades, chord_root, chord_tip,
                 twist_root_deg, twist_tip_deg,
                 thick, hub_r=0.12, hub_h=0.20,
                 blade_len=0.82, sweep_z=0.12, n_sec=16):
    tris = []
    cylinder(tris, hub_r, hub_h, segs=40)
    for b in range(num_blades):
        blade(tris, num_blades, b,
              r0=hub_r, r1=hub_r + blade_len,
              n_sec=n_sec,
              chord_root=chord_root, chord_tip=chord_tip,
              twist_root=np.radians(twist_root_deg),
              twist_tip=np.radians(twist_tip_deg),
              thick=thick,
              sweep_z=sweep_z)
    return tris


if __name__ == '__main__':
    configs = [
        # (num_blades, chord_root, chord_tip, twist_root, twist_tip, thick, name)
        (3, 0.38, 0.10, 50, 18, 0.038, 'turbine-1'),  # 3-blade wide propeller
        (5, 0.28, 0.08, 45, 14, 0.032, 'turbine-2'),  # 5-blade medium
        (7, 0.20, 0.06, 40, 12, 0.026, 'turbine-3'),  # 7-blade narrow high-speed
    ]
    for num_blades, cr, ct, tr, tt, th, name in configs:
        tris = make_turbine(num_blades, cr, ct, tr, tt, th)
        path = os.path.join(OUT_DIR, name + '.stl')
        write_stl(path, tris)
        print(f'{name}: {len(tris)} triangles → {path}')
    print('Done.')
