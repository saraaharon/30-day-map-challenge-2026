"""
Day 27 - Focsani to the Mountains (Blender)
Run (Windows):
  "C:\\Program Files\\Blender Foundation\\Blender 4.2\\blender.exe" -b -P render_blender.py -- test   (4 quick preview frames)
  "C:\\Program Files\\Blender Foundation\\Blender 4.2\\blender.exe" -b -P render_blender.py           (full render)
Needs dem.npy + dem_meta.json in the same folder. Units: 1 Blender unit = 1 km.
"""
import bpy, os, sys, json, math
import numpy as np
from mathutils import Vector
import bmesh

# ---------------- settings you can tweak ----------------
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
TEST = "test" in ARGS
FPS, SECONDS = 30, 14
RES_X, RES_Y = 1080, 1350
EXAG = 4.0             # vertical exaggeration
STEP = 2               # use every 2nd DEM pixel (180 m grid)
ENGINE = "EEVEE"       # change to "CYCLES" if EEVEE gives errors
SUN_STRENGTH = 4.0
LINE_STRENGTH = 2.5

# camera keyframes: (camera position, look-at target) in km, Focsani = (0, 0)
# x = east, y = north, z = up (already exaggerated)
CAM = np.array([[2, -1.5, 0.6], [-14, -9, 2.8], [-36, -22, 7.5], [-52, -38, 15.0]], float)
TGT = np.array([[-5, 2, 0.6], [-40, 4, 2.0], [-62, 8, 3.5], [-70, 10, 3.0]], float)

# ---------------- load DEM ----------------
h_full = np.load(os.path.join(HERE, "dem.npy")).astype("float32")
h_full[h_full < 0] = 0
meta = json.load(open(os.path.join(HERE, "dem_meta.json")))
px_km = meta["pixel_m"] / 1000.0
fr, fc = meta["focsani_row"], meta["focsani_col"]
z0 = float(h_full[fr, fc]) / 1000.0 * EXAG

# ---------------- start at the obelisk (Piata Unirii) ----------------
OB_LAT, OB_LON = 45.695911, 27.184700     # <-- paste your coordinates here
LON0, LAT0 = 27.1786, 45.6967
ox = (OB_LON - LON0) * 111.32 * math.cos(math.radians(LAT0))
oy = (OB_LAT - LAT0) * 111.13
orow = int(np.clip(fr - oy / px_km, 0, h_full.shape[0] - 1))
ocol = int(np.clip(fc + ox / px_km, 0, h_full.shape[1] - 1))
oz = float(h_full[orow, ocol]) / 1000.0 * EXAG

# new first keyframes: low and close, looking at the obelisk
CAM = np.vstack([[ox + 0.10, oy - 0.14, oz + 0.07],
                 [ox - 0.5, oy - 0.9, oz + 0.35], CAM])
TGT = np.vstack([[ox, oy, oz + 0.04],
                 [ox - 0.4, oy + 0.2, oz + 0.15], TGT])

h = h_full[::STEP, ::STEP]
rows, cols = h.shape
xs = (np.arange(cols) * STEP - fc) * px_km
ys = -(np.arange(rows) * STEP - fr) * px_km
X, Y = np.meshgrid(xs, ys)
Z = h / 1000.0 * EXAG
verts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
idx = np.arange(rows * cols).reshape(rows, cols)
quads = np.column_stack([idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(),
                         idx[1:, 1:].ravel(), idx[1:, :-1].ravel()])

# ---------------- scene ----------------
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

me = bpy.data.meshes.new("terrain")
me.from_pydata(verts.tolist(), [], quads.tolist())
me.update()
me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
terrain = bpy.data.objects.new("terrain", me)
scene.collection.objects.link(terrain)


def terrain_material():
    m = bpy.data.materials.new("neon_terrain")
    m.use_nodes = True
    nt = m.node_tree
    N, L = nt.nodes, nt.links
    N.clear()

    def MX(op, a, b=None):
        n = N.new("ShaderNodeMath")
        n.operation = op
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                n.inputs[i].default_value = v
            else:
                L.new(v, n.inputs[i])
        return n.outputs[0]

    out = N.new("ShaderNodeOutputMaterial")
    geo = N.new("ShaderNodeNewGeometry")
    sep = N.new("ShaderNodeSeparateXYZ")
    L.new(geo.outputs["Position"], sep.inputs["Vector"])
    x, y, z = sep.outputs["X"], sep.outputs["Y"], sep.outputs["Z"]

    # contour lines every 100 m
    contour = MX("LESS_THAN", MX("FRACT", MX("MULTIPLY", z, 10.0 / EXAG)), 0.07)
    # grid lines every 5 km
    gx = MX("LESS_THAN", MX("FRACT", MX("MULTIPLY", x, 1 / 5.0)), 0.035)
    gy = MX("LESS_THAN", MX("FRACT", MX("MULTIPLY", y, 1 / 5.0)), 0.035)
    grid = MX("MAXIMUM", gx, gy)

    # colour by height: pink (plain) -> purple -> cyan (peaks)
    hn = MX("MULTIPLY", z, 1.0 / (1.8 * EXAG))
    ramp = N.new("ShaderNodeValToRGB")
    L.new(hn, ramp.inputs["Fac"])
    stops = [(0.0, (1.0, 0.10, 0.55, 1)), (0.3, (0.55, 0.15, 1.0, 1)),
             (0.65, (0.0, 0.85, 1.0, 1)), (1.0, (0.85, 1.0, 1.0, 1))]
    cr = ramp.color_ramp
    cr.elements[0].position, cr.elements[0].color = stops[0]
    cr.elements[1].position, cr.elements[1].color = stops[-1]
    for p, c in stops[1:-1]:
        cr.elements.new(p).color = c

    em1 = N.new("ShaderNodeEmission")
    L.new(ramp.outputs["Color"], em1.inputs["Color"])
    L.new(MX("ADD", MX("MULTIPLY", contour, LINE_STRENGTH), 0.10), em1.inputs["Strength"])

    em2 = N.new("ShaderNodeEmission")
    em2.inputs["Color"].default_value = (1.0, 0.15, 0.7, 1)
    L.new(MX("MULTIPLY", grid, 1.2), em2.inputs["Strength"])

    base = N.new("ShaderNodeBsdfPrincipled")
    base.inputs["Base Color"].default_value = (0.04, 0.02, 0.10, 1)
    base.inputs["Roughness"].default_value = 0.9

    a1 = N.new("ShaderNodeAddShader")
    a2 = N.new("ShaderNodeAddShader")
    L.new(base.outputs[0], a1.inputs[0])
    L.new(em1.outputs[0], a1.inputs[1])
    L.new(a1.outputs[0], a2.inputs[0])
    L.new(em2.outputs[0], a2.inputs[1])
    L.new(a2.outputs[0], out.inputs["Surface"])
    return m


def emissive(name, color, strength):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    N = m.node_tree.nodes
    N.clear()
    em = N.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = color
    em.inputs["Strength"].default_value = strength
    out = N.new("ShaderNodeOutputMaterial")
    m.node_tree.links.new(em.outputs[0], out.inputs["Surface"])
    return m, em


terrain.data.materials.append(terrain_material())


# ---------------- gold x-ray buildings ----------------
BEXAG = 3   # building height exaggeration (raise it if the city looks flat from afar)
bj = json.load(open(os.path.join(HERE, "buildings.json")))["b"]

bm = bmesh.new()
hl = bm.verts.layers.float.new("h")            # height in metres, drives the colour ramp
for hm, flat in bj:
    pts = list(zip(flat[0::2], flat[1::2]))
    if len(pts) < 3:
        continue
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    r = int(np.clip(fr - cy / px_km, 0, h_full.shape[0] - 1))
    c = int(np.clip(fc + cx / px_km, 0, h_full.shape[1] - 1))
    zb = float(h_full[r, c]) / 1000.0 * EXAG           # sit on the terrain
    zt = zb + hm / 1000.0 * BEXAG
    bot = [bm.verts.new((x, y, zb)) for x, y in pts]
    top = [bm.verts.new((x, y, zt)) for x, y in pts]
    for v in bot + top:
        v[hl] = hm
    try:
        for i in range(len(pts)):
            j = (i + 1) % len(pts)
            bm.faces.new((bot[i], bot[j], top[j], top[i]))
        bm.faces.new(top)
    except ValueError:
        pass                                           # skip degenerate footprints
bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
bme = bpy.data.meshes.new("buildings")
bm.to_mesh(bme)
bm.free()


def gold_xray():
    m = bpy.data.materials.new("gold_xray")
    m.use_nodes = True
    for attr, val in (("surface_render_method", "BLENDED"), ("blend_method", "BLEND")):
        try:
            setattr(m, attr, val)
        except Exception:
            pass
    nt = m.node_tree
    N, L = nt.nodes, nt.links
    N.clear()
    out = N.new("ShaderNodeOutputMaterial")
    at = N.new("ShaderNodeAttribute")
    at.attribute_name = "h"
    nh = N.new("ShaderNodeMath")                        # metres -> 0..1 (40 m = full)
    nh.operation = "MULTIPLY"
    nh.inputs[1].default_value = 1 / 60.0
    L.new(at.outputs["Fac"], nh.inputs[0])
    ramp = N.new("ShaderNodeValToRGB")
    L.new(nh.outputs[0], ramp.inputs["Fac"])
    cr = ramp.color_ramp
    cr.elements[0].color = (0.9, 0.30, 0.0, 1)        # amber (low)
    cr.elements[1].color = (1.0, 0.75, 0.25, 1)        # white-gold (tall)
    cr.elements[1].position = 1.0
    mid = cr.elements.new(0.5)
    mid.color = (1.0, 0.55, 0.05, 1)                   # gold
    lw = N.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.3
    st = N.new("ShaderNodeMath")                        # glow stronger at silhouettes
    st.operation = "MULTIPLY_ADD"
    L.new(lw.outputs["Fresnel"], st.inputs[0])
    st.inputs[1].default_value = 1.2
    st.inputs[2].default_value = 0.25
    fa = N.new("ShaderNodeMath")                        # opacity: faint centre, solid edges
    fa.operation = "MULTIPLY_ADD"
    L.new(lw.outputs["Fresnel"], fa.inputs[0])
    fa.inputs[1].default_value = 0.85
    fa.inputs[2].default_value = 0.15
    em = N.new("ShaderNodeEmission")
    L.new(ramp.outputs["Color"], em.inputs["Color"])
    L.new(st.outputs[0], em.inputs["Strength"])
    tr = N.new("ShaderNodeBsdfTransparent")
    mx = N.new("ShaderNodeMixShader")
    L.new(fa.outputs[0], mx.inputs["Fac"])
    L.new(tr.outputs[0], mx.inputs[1])
    L.new(em.outputs[0], mx.inputs[2])
    L.new(mx.outputs[0], out.inputs["Surface"])
    return m


bld = bpy.data.objects.new("buildings", bme)
scene.collection.objects.link(bld)
bld.data.materials.append(gold_xray())

# gold wireframe edges on a second object sharing the same mesh
edges_m, _ = emissive("gold_edges", (1.0, 0.65, 0.1, 1), 1.5)
bld_e = bpy.data.objects.new("buildings_edges", bme)
scene.collection.objects.link(bld_e)
wf = bld_e.modifiers.new("wire", "WIREFRAME")
wf.thickness = 0.002
wf.use_replace = True
wf.material_offset = 1      # wireframe uses slot 1 (gold_edges)
bme.materials.append(edges_m)

# ---------------- world, light, camera ----------------
world = bpy.data.worlds.new("world")
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs[0].default_value = (0.015, 0.008, 0.04, 1)
bg.inputs[1].default_value = 1.0
scene.world = world

ld = bpy.data.lights.new("sun", "SUN")
ld.energy = SUN_STRENGTH
ld.color = (0.75, 0.6, 1.0)
sun = bpy.data.objects.new("sun", ld)
scene.collection.objects.link(sun)
sun.rotation_euler = (0, math.radians(65), math.radians(20))  # low sun from the east

cam_data = bpy.data.cameras.new("cam")
cam_data.lens = 30
cam_data.clip_end = 2000
cam_data.clip_start = 0.005    # default is 100 m, which would clip anything this close

cam = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam
target = bpy.data.objects.new("target", None)
scene.collection.objects.link(target)
con = cam.constraints.new("TRACK_TO")
con.target = target
con.track_axis = "TRACK_NEGATIVE_Z"
con.up_axis = "UP_Y"

# ---------------- render settings ----------------
scene.render.resolution_x = RES_X
scene.render.resolution_y = RES_Y
scene.render.resolution_percentage = 50 if TEST else 100
scene.render.image_settings.file_format = "PNG"

if ENGINE == "CYCLES":
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 24
    scene.cycles.use_denoising = True
else:
    for name in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            scene.render.engine = name
            break
        except TypeError:
            pass
    try:
        scene.eevee.taa_render_samples = 48
    except Exception:
        pass

try:
    scene.view_settings.view_transform = "Standard"
except Exception:
    pass

# glow via compositor (skipped automatically if this Blender version differs)
try:
    if True: raise Exception("glow disabled")
    if hasattr(scene, "compositing_node_group"):          # Blender 5.x          # Blender 5.x
        tree = bpy.data.node_groups.new("comp", "CompositorNodeTree")
        scene.compositing_node_group = tree
        tree.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        nodes, links = tree.nodes, tree.links
        outn = nodes.new("NodeGroupOutput")
    else:                                                 # Blender 4.x
        scene.use_nodes = True
        tree = scene.node_tree
        nodes, links = tree.nodes, tree.links
        for n in list(nodes):
            nodes.remove(n)
        outn = nodes.new("CompositorNodeComposite")
    rl = nodes.new("CompositorNodeRLayers")
    gl = nodes.new("CompositorNodeGlare")
    gl.glare_type = "FOG_GLOW"
    for key, val in (("Threshold", 1.0), ("Size", 0.5), ("Quality", "High")):
        if key in gl.inputs:
            try: gl.inputs[key].default_value = val
            except Exception: pass
        else:
            try: setattr(gl, key.lower(), 7 if key == "Size" else val)
            except Exception: pass
    links.new(rl.outputs["Image"], gl.inputs["Image"])
    links.new(gl.outputs["Image"], outn.inputs[0])
    print("Glow OK")
except Exception as e:
    print("Glow skipped:", e)


# ---------------- camera path ----------------
def catmull(pts, s):
    n = len(pts)
    i = min(int(s), n - 2)
    t = s - i
    p0, p1, p2, p3 = pts[max(i - 1, 0)], pts[i], pts[i + 1], pts[min(i + 2, n - 1)]
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)


def place(u, frame_no):
    e = u * u * u * (u * (6 * u - 15) + 10)  # smootherstep ease in/out
    s = e * (len(CAM) - 1)
    cam.location = Vector(catmull(CAM, s))
    target.location = Vector(catmull(TGT, s))
    bpy.context.view_layer.update()


N_FRAMES = FPS * SECONDS
if TEST:
    out_dir = os.path.join(HERE, "frames_test")
    jobs = [(k, round(u * (N_FRAMES - 1)), u) for k, u in enumerate((0, 0.33, 0.66, 1.0))]
else:
    out_dir = os.path.join(HERE, "frames")
    jobs = [(f, f, f / (N_FRAMES - 1)) for f in range(N_FRAMES)]
os.makedirs(out_dir, exist_ok=True)

for k, f, u in jobs:
    place(u, f)
    scene.render.filepath = os.path.join(out_dir, f"frame_{k:04d}.png")
    bpy.ops.render.render(write_still=True)
    print(f"rendered {k + 1}/{len(jobs)}")
print("DONE ->", out_dir)
