"""
touch_cylinder.py — Isaac Sim 5.1 + cuRobo single-object touch pipeline.

Touches a single object (the cylinder) and records its tactile squeeze
signature, no lift. Straight-line motion logic (CASE13_WEIGHT holds
orientation + X + Y, frees Z).

Sequence:
  spawn at INITIAL_JOINTS_RAD
  -> free move (tool-position IK) to Point_cylinder_up   (orientation held)
  -> straight DOWN to Point_cylinder_grasp               (Z-only)
       [RECORD ON] close to GRIPPER_CYLINDER -> hold 1s -> open  [RECORD OFF]
  -> straight UP to Point_cylinder_up -> hold.

Recording starts when the gripper begins to close and stops the instant it is
fully open again. One CSV PAIR (LEFT + RIGHT), each row tagged with `object`
(cylinder) and `phase` (closing/holding/opening). At the end a gripper frame
timeline is printed to the console.

# NOTE: all deformable-material and physics tuning has been removed. The scene
# runs with its authored deformable material, masses, friction, contact
# offsets, attachment damping, solver iterations and collisions. The only
# physics setting still applied at runtime is GPU dynamics enablement (required
# for deformable bodies to run at all).

"""

# ---------------------------------------------------------------------------
# 1) Launch Kit
# ---------------------------------------------------------------------------
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False, "physics_gpu": 0})

import numpy as np, carb, torch, time, datetime, os, csv as csvlib
carb.settings.get_settings().set("/physics/enableDeformableBodies", True)
carb.settings.get_settings().set("/physics/enableGpuDynamics", True)
# TSF_85_Ext record gate starts OFF: the extension writes NO CSV rows until
# this is flipped True (done just before the gripper close, flipped back False
# after the open). Frame numbers written are the real simulation frames.
carb.settings.get_settings().set("/exts/TSF_85_Ext/record_active", False)

from isaacsim.core.api import World
from isaacsim.core.api.objects import VisualSphere
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from pxr import UsdPhysics, PhysxSchema, Usd, UsdGeom, UsdShade, Sdf

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState
from curobo.util_file import load_yaml
from curobo.wrap.reacher.motion_gen import (
    MotionGen, MotionGenConfig, MotionGenPlanConfig, PoseCostMetric)

# ---------------------------------------------------------------------------
# 2) Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
SCENES_DIR  = os.path.join(SCRIPT_DIR, "scenes")
DATA_DIR    = os.path.join(SCRIPT_DIR, "data_generated")

USD_PATH = os.path.join(SCENES_DIR, "scene_cylinder.usd")
CUROBO_ROBOT_YAML = os.path.join(SCENES_DIR, "ur5e.yml")
ROBOT_PRIM_PATH = "/World/robot_gripper_adapter_sensor"
OUTPUT_TXT = os.path.join(DATA_DIR, "motion_points_touch_cylinder.txt")
os.makedirs(DATA_DIR, exist_ok=True)

# Auto-detected on load (fallback values here).
ROBOT_WORLD_POS = np.array([0.0, -0.3375, 0.99275])
ROBOT_WORLD_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])

ARM_JOINT_NAMES = ["shoulder_pan_joint","shoulder_lift_joint","elbow_joint",
                   "wrist_1_joint","wrist_2_joint","wrist_3_joint"]

# --- INITIAL pose from the REAL robot --------------------------------------
# Six joint values (radians, in ARM_JOINT_NAMES order). The robot is spawned
# directly at this configuration so the sim starts in the reported pose.
INITIAL_JOINTS_RAD = np.array([
    -0.992425,   # J_shoulder_pan_rad
    -2.179929,   # J_shoulder_lift_rad
    -0.865866,   # J_elbow_rad
    -1.667783,   # J_wrist_1_rad
     1.570776,   # J_wrist_2_rad
    -0.992413,   # J_wrist_3_rad
])
pos0_rad = INITIAL_JOINTS_RAD

# ===========================================================================
# WAYPOINTS — all in WORLD frame (converted to base/cuRobo frame at runtime).
# Each object has an "up" point (approach/retreat height) and a "grasp" point
# (straight down from up). Edit freely.
# ===========================================================================
POINT_CYLINDER_UP_WORLD    = np.array([-0.26806, 0.199, 1.34244])
POINT_CYLINDER_GRASP_WORLD = np.array([-0.26806, 0.199, 1.24244])


TOOL_DOWN_ROTVEC = np.array([2.2214, 2.2214, 0.0])

# Stitched-straight settings (used for every straight Z up/down move).
N_STEPS = 10
CASE13_WEIGHT = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]  # hold orient+X+Y, free Z

# Time-dilation factor for the straight Z moves. cuRobo convention: <1 slows
# the trajectory down (longer duration, lower velocities & accelerations),
# >1 speeds it up. Descent (up->grasp) and ascent (grasp->up) each have their
# own factor so you can slow them independently.
DESCENT_TIME_DILATION = 1.0      # up -> grasp straight descent
ASCENT_TIME_DILATION  = 1.0      # grasp -> up straight ascent

# Gripper.
GRIPPER_DRIVE_JOINT = "finger_joint"
GRIPPER_OPEN        = 0.0            # fully open (release)
# Per-object grasp (close) targets, in radians. Currently all 0.54 for
# testing; will be given different values later (different objects/shapes).
GRIPPER_CYLINDER = 0.55
# Number of physics frames (at 120 Hz) over which the close OR open ramp is
# spread. Bigger = slower gripper motion. 60 frames = 0.5 s; 240 = 2 s;
# 600 = 5 s. Default 60 (same speed as the original pipeline); raise to slow
# the close/open down.
GRIPPER_RAMP_FRAMES = 60
WAIT_GRASP_SECONDS  = 1.0            # settle at grasp point before closing
WAIT_HOLD_SECONDS   = 1.0            # hold closed before opening

# --- Object lookup ----------------------------------------------------------
# The Object Xform is auto-located anywhere in the stage by name (used by the
# motion/recording only — no physics material is applied to it anymore; the
# scene's authored values are used as-is).
OBJECT_XFORM_NAME = "Object"

# Joint alignment diagnostic: scan every UsdPhysics.Joint and report any
# whose two bodies disagree on where the joint frame is. PhysX warns about
# these ("disjointed body transforms ... snap objects together") because it
# has to force-snap them at init, leaving residual stress in the system.
DIAGNOSE_JOINT_ALIGNMENT          = True
JOINT_ALIGNMENT_THRESHOLD_MM      = 0.5   # flag any joint with discrepancy > this

# --- Tactile sensor recording -----------------------------------------------
CSV_DIR      = DATA_DIR   # all CSVs go in data_generated/ alongside the report
CSV_BASENAME = "touch_cylinder"
LEFT_SENSOR_FILTER  = "left"
RIGHT_SENSOR_FILTER = "right"
CASE_NAME_HINT      = "case"

# ---------------------------------------------------------------------------
# 3) Helpers
# ---------------------------------------------------------------------------
def rotvec_to_quat(rv):
    a = float(np.linalg.norm(rv))
    if a < 1e-9: return np.array([1.,0,0,0])
    ax = rv/a; s = np.sin(a/2)
    return np.array([np.cos(a/2), ax[0]*s, ax[1]*s, ax[2]*s])

def rotmat(q):
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def world_to_base(p):
    return rotmat(ROBOT_WORLD_QUAT_WXYZ).T @ (p - ROBOT_WORLD_POS)

def base_to_world(p):
    return ROBOT_WORLD_POS + rotmat(ROBOT_WORLD_QUAT_WXYZ) @ p


def _joint_attr(joint_prim, name, default):
    a = joint_prim.GetAttribute(name)
    if not a or not a.IsValid() or a.Get() is None:
        return default
    return a.Get()

def _quat_to_wxyz(q):
    if q is None: return np.array([1.0, 0.0, 0.0, 0.0])
    return np.array([float(q.GetReal()),
                     float(q.GetImaginary()[0]),
                     float(q.GetImaginary()[1]),
                     float(q.GetImaginary()[2])])

def diagnose_one_joint(stage, joint_prim, verbose=False):
    """Compute the world-space position of the joint frame as seen from each
    body. Return discrepancy magnitude in meters (None if undeterminable)."""
    joint = UsdPhysics.Joint(joint_prim)
    b0_targets = joint.GetBody0Rel().GetTargets()
    b1_targets = joint.GetBody1Rel().GetTargets()
    # A joint with only one body is anchored to world — no alignment check needed
    if not b0_targets or not b1_targets:
        if verbose:
            print(f"[joint-diag] {joint_prim.GetPath()}: single-body joint, "
                  f"body0={b0_targets} body1={b1_targets} (anchored)")
        return None

    b0_prim = stage.GetPrimAtPath(b0_targets[0])
    b1_prim = stage.GetPrimAtPath(b1_targets[0])
    if not b0_prim.IsValid() or not b1_prim.IsValid():
        return None

    lp0 = np.array(_joint_attr(joint_prim, "physics:localPos0", (0.0, 0.0, 0.0)),
                   dtype=float)
    lp1 = np.array(_joint_attr(joint_prim, "physics:localPos1", (0.0, 0.0, 0.0)),
                   dtype=float)
    lr0 = _quat_to_wxyz(_joint_attr(joint_prim, "physics:localRot0", None))
    lr1 = _quat_to_wxyz(_joint_attr(joint_prim, "physics:localRot1", None))

    (b0_pos, b0_quat) = get_world_pose(b0_prim)
    (b1_pos, b1_quat) = get_world_pose(b1_prim)
    b0_pos = np.array(b0_pos); b1_pos = np.array(b1_pos)
    b0_quat = np.array(b0_quat); b1_quat = np.array(b1_quat)

    # Joint frame world position as derived from each body.
    joint_w_from_b0 = b0_pos + rotmat(b0_quat) @ lp0
    joint_w_from_b1 = b1_pos + rotmat(b1_quat) @ lp1
    diff = joint_w_from_b1 - joint_w_from_b0
    diff_mag = float(np.linalg.norm(diff))

    if verbose:
        print(f"[joint-diag] {joint_prim.GetPath()}")
        print(f"               body0 ({b0_targets[0]}): world pos={b0_pos}")
        print(f"                     local frame pos={lp0}  rot(wxyz)={lr0}")
        print(f"               body1 ({b1_targets[0]}): world pos={b1_pos}")
        print(f"                     local frame pos={lp1}  rot(wxyz)={lr1}")
        print(f"               joint world pos via body0: {joint_w_from_b0}")
        print(f"               joint world pos via body1: {joint_w_from_b1}")
        print(f"               POSITION DISCREPANCY: {diff}  |diff|={diff_mag*1000:.3f} mm")
    return diff_mag

def diagnose_all_joints(stage, threshold_mm=0.5):
    """Scan every UsdPhysics.Joint and flag those whose two bodies disagree
    on the joint frame by more than `threshold_mm`. Prints a summary plus a
    verbose breakdown of any flagged joint."""
    threshold_m = threshold_mm / 1000.0
    all_joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
    print(f"[joint-diag] Scanning {len(all_joints)} joints "
          f"(threshold = {threshold_mm} mm)...")
    flagged = []
    for jp in all_joints:
        mag = diagnose_one_joint(stage, jp, verbose=False)
        if mag is not None and mag > threshold_m:
            flagged.append((jp, mag))
    if not flagged:
        print(f"[joint-diag] No joints exceed {threshold_mm} mm — all aligned.")
        return
    flagged.sort(key=lambda t: -t[1])
    print(f"[joint-diag] {len(flagged)} joints exceed {threshold_mm} mm:")
    for jp, mag in flagged:
        print(f"[joint-diag]   {mag*1000:8.3f} mm   {jp.GetPath()}")
    # Verbose breakdown of the worst offender.
    worst_jp, worst_mag = flagged[0]
    print(f"[joint-diag] Verbose breakdown of worst joint "
          f"({worst_mag*1000:.3f} mm):")
    diagnose_one_joint(stage, worst_jp, verbose=True)

# --- Deformable-mesh + Case-pose recording helpers --------------------------
def _has_physx_deformable_api(prim):
    if prim is None: return False
    try:
        if prim.HasAPI(PhysxSchema.PhysxDeformableBodyAPI): return True
    except Exception: pass
    schemas = prim.GetAppliedSchemas()
    return any(s in schemas for s in (
        "OmniPhysicsDeformableBodyAPI",
        "OmniPhysicsVolumeDeformableSimAPI",
        "PhysxAutoDeformableBodyAPI",
    ))

def discover_deformable_meshes(stage, root_path):
    out = []
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid(): return out
    for p in Usd.PrimRange(root):
        if not _has_physx_deformable_api(p): continue
        mesh = p
        for c in p.GetChildren():
            if "simulation_mesh" in c.GetName().lower():
                mesh = c; break
        out.append(mesh)
    return out

def _get_deformable_attrs(mesh_prim):
    def _first(*names):
        for n in names:
            a = mesh_prim.GetAttribute(n)
            if a and a.IsValid(): return a
        return None
    pos  = _first("physxDeformable:simulationPoints", "points")
    rest = _first("physxDeformable:simulationRestPoints",
                  "omniphysics:restShapePoints", "restPoints")
    vel  = _first("physxDeformable:simulationVelocities", "velocities")
    return pos, rest, vel

def find_case_prim(stage, root_path, side_filter, name_hint=CASE_NAME_HINT):
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid(): return None
    cands = []
    for p in Usd.PrimRange(root):
        if side_filter.lower() in str(p.GetPath()).lower() \
           and name_hint.lower() in p.GetName().lower():
            cands.append(p)
    if not cands: return None
    if len(cands) > 1:
        print(f"[warn] Multiple '{name_hint}' prims match side='{side_filter}':")
        for c in cands: print(f"        {c.GetPath()}")
        print(f"[warn] Using the first.")
    return cands[0]

def get_world_pose(prim):
    """World-frame pose via the full prim ancestry. Available but not used
    for recording (we use local xformOp values to match the Property panel)."""
    xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t  = xf.ExtractTranslation()
    q  = xf.ExtractRotation().GetQuat()
    im = q.GetImaginary()
    return ((float(t[0]), float(t[1]), float(t[2])),
            (float(q.GetReal()), float(im[0]), float(im[1]), float(im[2])))

def get_local_xformop_pose(prim):
    """Raw `xformOp:translate` and `xformOp:orient` — matches the Isaac Sim
    Property panel and the Lateral1.py convention."""
    tx = ty = tz = 0.0
    ow, ox, oy, oz = 1.0, 0.0, 0.0, 0.0
    if prim is None:
        return (tx, ty, tz), (ow, ox, oy, oz)
    t_attr = prim.GetAttribute("xformOp:translate")
    o_attr = prim.GetAttribute("xformOp:orient")
    if t_attr and t_attr.IsValid():
        t = t_attr.Get()
        if t is not None:
            tx, ty, tz = float(t[0]), float(t[1]), float(t[2])
    if o_attr and o_attr.IsValid():
        o = o_attr.Get()
        if o is not None and hasattr(o, "GetReal"):
            ow = float(o.GetReal())
            im = o.GetImaginary()
            ox, oy, oz = float(im[0]), float(im[1]), float(im[2])
    return (tx, ty, tz), (ow, ox, oy, oz)

class SensorRecorder:
    HEADER = [
        "frame", "t", "object", "phase", "node_id",
        "s1_x",  "s1_y",  "s1_z",
        "s1_vx", "s1_vy", "s1_vz",
        "s1_Rx", "s1_Ry", "s1_Rz",
        "s1_Trans_x", "s1_Trans_y", "s1_Trans_z",
        "s1_Ori_w", "s1_Ori_x", "s1_Ori_y", "s1_Ori_z",
    ]
    def __init__(self, name, mesh_prim, case_prim, csv_path):
        self.name = name
        self.mesh_prim = mesh_prim
        self.case_prim = case_prim
        self.pos_attr, self.rest_attr, self.vel_attr = _get_deformable_attrs(mesh_prim)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self.csv_file = open(csv_path, "w", newline="")
        self.csv_writer = csvlib.writer(self.csv_file, delimiter="\t")
        self.csv_writer.writerow(self.HEADER)
        self.csv_file.flush()

    def record(self, frame, sim_time, obj_label="", phase_label=""):
        if self.pos_attr is None: return
        P = self.pos_attr.Get()
        if P is None: return
        V = self.vel_attr.Get()  if (self.vel_attr  and self.vel_attr.IsValid())  else None
        R = self.rest_attr.Get() if (self.rest_attr and self.rest_attr.IsValid()) else None

        if self.case_prim is not None:
            (tx, ty, tz), (ow, ox, oy, oz) = get_local_xformop_pose(self.case_prim)
        else:
            tx = ty = tz = 0.0; ow, ox, oy, oz = 1.0, 0.0, 0.0, 0.0

        n   = len(P)
        n_v = len(V) if V is not None else 0
        n_r = len(R) if R is not None else 0
        for i in range(n):
            vx = vy = vz = 0.0
            if V is not None and i < n_v:
                vx, vy, vz = float(V[i][0]), float(V[i][1]), float(V[i][2])
            rx = ry = rz = 0.0
            if R is not None and i < n_r:
                rx, ry, rz = float(R[i][0]), float(R[i][1]), float(R[i][2])
            self.csv_writer.writerow([
                frame, f"{sim_time:.6f}", obj_label, phase_label, i,
                float(P[i][0]), float(P[i][1]), float(P[i][2]),
                vx, vy, vz,
                rx, ry, rz,
                tx, ty, tz,
                ow, ox, oy, oz,
            ])
        if frame % 10 == 0:
            self.csv_file.flush()

    def close(self):
        try:
            self.csv_file.flush(); self.csv_file.close()
        except Exception as e:
            print(f"[CSV][{self.name}] close failed: {e}")

def build_recorders(stage, suffix=""):
    """Build a LEFT/RIGHT recorder pair. `suffix` is appended to the basename
    so each object gets its own CSV pair, e.g. grasp_recog_obj1_LEFT_<ts>.csv.
    Returns (left_rec, right_rec, left_csv_path, right_csv_path)."""
    all_def = discover_deformable_meshes(stage, ROBOT_PRIM_PATH)

    def match(side, kind):
        cands = [d for d in all_def if side.lower() in str(d.GetPath()).lower()]
        if not cands:
            raise RuntimeError(
                f"No {kind} deformable mesh path contains '{side}'. "
                f"Adjust LEFT/RIGHT_SENSOR_FILTER. "
                f"Available: {[str(d.GetPath()) for d in all_def]}")
        return cands[0]

    left_mesh  = match(LEFT_SENSOR_FILTER,  "left sensor")
    right_mesh = match(RIGHT_SENSOR_FILTER, "right sensor")
    left_case  = find_case_prim(stage, ROBOT_PRIM_PATH, LEFT_SENSOR_FILTER)
    right_case = find_case_prim(stage, ROBOT_PRIM_PATH, RIGHT_SENSOR_FILTER)
    if left_case  is None: print("[warn] No 'Case' prim found for LEFT  side — Trans/Ori will be zeros.")
    if right_case is None: print("[warn] No 'Case' prim found for RIGHT side — Trans/Ori will be zeros.")

    sfx = f"_{suffix}" if suffix else ""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    left_csv  = os.path.join(CSV_DIR, f"{CSV_BASENAME}{sfx}_LEFT_{ts}.csv")
    right_csv = os.path.join(CSV_DIR, f"{CSV_BASENAME}{sfx}_RIGHT_{ts}.csv")
    return (SensorRecorder("LEFT",  left_mesh,  left_case,  left_csv),
            SensorRecorder("RIGHT", right_mesh, right_case, right_csv),
            left_csv, right_csv)

# ---------------------------------------------------------------------------
# 4) World + USD
# ---------------------------------------------------------------------------
world = World(stage_units_in_meters=1.0, physics_dt=1/120., rendering_dt=1/60., backend="numpy")
pc = world.get_physics_context(); pc.enable_gpu_dynamics(True); pc.set_broadphase_type("GPU")
world.scene.add_default_ground_plane()
print(f"[scene] loading USD: {USD_PATH}")
print(f"[data]  output folder: {DATA_DIR}")
add_reference_to_stage(usd_path=USD_PATH, prim_path=ROBOT_PRIM_PATH)
stage = world.stage
# Enable GPU dynamics on every PhysicsScene (REQUIRED — deformable bodies only
# run on GPU). 

for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.Scene):
        a = PhysxSchema.PhysxSceneAPI.Apply(prim)
        a.CreateEnableGPUDynamicsAttr().Set(True)
        try: a.CreateBroadphaseTypeAttr().Set("GPU")
        except Exception: pass
        print(f"[scene] GPU dynamics enabled on {prim.GetPath()}")

# Joint-alignment diagnostic — flags joints PhysX is force-snapping.
if DIAGNOSE_JOINT_ALIGNMENT:
    diagnose_all_joints(stage, threshold_mm=JOINT_ALIGNMENT_THRESHOLD_MM)

# ---------------------------------------------------------------------------
# 4b) Enable the TSF_85_Ext tactile extension (headless) so IT writes the
#     three CSV files per sensor (TactileData_s1_* and TactileData_s2_*).
#     The extension reads these carb settings on startup. Sensor roots are
#     given in their authored (non-doubled) form; the extension's runtime
#     path resolver corrects for reference-nesting automatically.
#     record_active stays False here — it is flipped True just before the
#     gripper close and back to False after the open (see grasp routine).
# ---------------------------------------------------------------------------
SENSOR_ROOT_RIGHT = f"{ROBOT_PRIM_PATH}/TSF_85_right/TSF_85"   # sensor 1 (s1, RIGHT)
SENSOR_ROOT_LEFT  = f"{ROBOT_PRIM_PATH}/TSF_85_left/TSF_85"    # sensor 2 (s2, LEFT)

_tsf = carb.settings.get_settings()
_tsf.set("/exts/TSF_85_Ext/headless",      True)
_tsf.set("/exts/TSF_85_Ext/sensor_root",   SENSOR_ROOT_RIGHT)
_tsf.set("/exts/TSF_85_Ext/sensor_root_2", SENSOR_ROOT_LEFT)
_tsf.set("/exts/TSF_85_Ext/output_dir",    DATA_DIR)
_tsf.set("/exts/TSF_85_Ext/base_name",     "TactileData")
_tsf.set("/exts/TSF_85_Ext/log_dz",        True)
_tsf.set("/exts/TSF_85_Ext/log_pred",      True)
_tsf.set("/exts/TSF_85_Ext/log_mesh",      True)

from omni.kit.app import get_app
_ext_mgr = get_app().get_extension_manager()
_ext_mgr.add_path(os.path.dirname(SCRIPT_DIR))   # parent folder that holds TSF_85_Ext/
_ok = _ext_mgr.set_extension_enabled_immediate("TSF_85_Ext", True)
print(f"[TSF85] Extension enabled={_ok}. Right=_s1, Left=_s2. Output: {DATA_DIR}")

# ---------------------------------------------------------------------------
# 5) Find UR5e articulation + auto base
# ---------------------------------------------------------------------------
def find_roots(s,u):
    rp=s.GetPrimAtPath(u)
    return [p for p in Usd.PrimRange(rp) if "PhysicsArticulationRootAPI" in p.GetAppliedSchemas()] if rp.IsValid() else []
def sub(p): return p.GetParent() if p.IsA(UsdPhysics.Joint) else p
def find_ur5e(s,u,jn):
    for c in find_roots(s,u):
        for x in Usd.PrimRange(sub(c)):
            if x.IsA(UsdPhysics.Joint):
                n=x.GetName()
                if any(n==j or n.endswith("/"+j) for j in jn): return c
    return None
ar = find_ur5e(stage, ROBOT_PRIM_PATH, ARM_JOINT_NAMES)
AP = str(ar.GetParent().GetPath()) if (ar and ar.IsA(UsdPhysics.Joint)) else (str(ar.GetPath()) if ar else ROBOT_PRIM_PATH)
print(f"[scene] Articulation: {AP}")

def find_base_link(stage, art_path):
    root = stage.GetPrimAtPath(art_path)
    if root.IsValid():
        for p in Usd.PrimRange(root):
            if p.GetName() == "base_link":
                return p
    return None

base_prim = find_base_link(stage, AP)
if base_prim is not None:
    xfc = UsdGeom.XformCache(Usd.TimeCode.Default())
    xf = xfc.GetLocalToWorldTransform(base_prim)
    t = xf.ExtractTranslation()
    q = xf.ExtractRotationQuat()
    ROBOT_WORLD_POS = np.array([t[0], t[1], t[2]])
    ROBOT_WORLD_QUAT_WXYZ = np.array([
        q.GetReal(),
        q.GetImaginary()[0], q.GetImaginary()[1], q.GetImaginary()[2]])
    print(f"[scene] AUTO base_link WORLD pos:  {ROBOT_WORLD_POS}")
    print(f"[scene] AUTO base_link WORLD quat: {ROBOT_WORLD_QUAT_WXYZ}")
else:
    print(f"[scene] WARNING: base_link not found under {AP}; "
          f"using fallback base pose {ROBOT_WORLD_POS}")

# Convert obj1 world waypoints into base/cuRobo frame.
cylinder_up_base    = world_to_base(POINT_CYLINDER_UP_WORLD)
cylinder_grasp_base = world_to_base(POINT_CYLINDER_GRASP_WORLD)

# Visual goal spheres for obj1: up point lighter, grasp point darker.
_GOALS = [
    ("Cylinder_up",    POINT_CYLINDER_UP_WORLD,    np.array([1.0, 1.0, 0.4])),  # light yellow
    ("Cylinder_grasp", POINT_CYLINDER_GRASP_WORLD, np.array([0.8, 0.8, 0.0])),  # dark yellow
]
for _gname, _gpos, _gcol in _GOALS:
    world.scene.add(VisualSphere(prim_path=f"/World/Goals/{_gname}",
        name=f"g_{_gname.lower()}", position=_gpos, radius=0.015, color=_gcol))
    print(f"[scene] Goal {_gname} world {_gpos} -> base {world_to_base(_gpos)}")

robot = SingleArticulation(prim_path=AP, name="ur5e"); world.scene.add(robot); world.reset()
dn = robot.dof_names
def idxs(dn,bn):
    o=[]
    for nm in bn:
        if nm in dn: o.append(dn.index(nm))
        else:
            c=[d for d in dn if d==nm or d.endswith("/"+nm) or d.endswith(nm)]; o.append(dn.index(c[0]))
    return np.array(o,dtype=np.int32)
ai = idxs(dn, ARM_JOINT_NAMES)

try:
    gi = np.array([dn.index(GRIPPER_DRIVE_JOINT)], dtype=np.int32)
except ValueError:
    cand=[d for d in dn if d.endswith("/"+GRIPPER_DRIVE_JOINT) or d.endswith(GRIPPER_DRIVE_JOINT)]
    gi = np.array([dn.index(cand[0])], dtype=np.int32) if cand else None

dp = np.array(robot.get_joint_positions(), dtype=np.float32); dp[ai]=pos0_rad
robot.set_joints_default_state(positions=dp)
robot.set_joint_positions(pos0_rad, joint_indices=ai)
robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=pos0_rad, joint_indices=ai))
for _ in range(10): world.step(render=True)

initial_q = robot.get_joint_positions()[ai].copy()

# ---------------------------------------------------------------------------
# 6) cuRobo
# ---------------------------------------------------------------------------
print("[curobo] loading...")
ta = TensorDeviceType()
rc = RobotConfig.from_dict(load_yaml(CUROBO_ROBOT_YAML)["robot_cfg"], ta)
mg = MotionGen(MotionGenConfig.load_from_robot_config(
    rc, world_model=None, tensor_args=ta, interpolation_dt=0.02,
    num_trajopt_seeds=4, project_pose_to_goal_frame=True, use_cuda_graph=False))
mg.warmup(enable_graph=False, warmup_js_trajopt=False)
print("[curobo] ready.")

def fk(q):
    qt=ta.to_device(q.astype(np.float32)).view(1,-1)
    f=mg.compute_kinematics(JointState.from_position(qt, joint_names=ARM_JOINT_NAMES))
    if hasattr(f,"ee_pose") and f.ee_pose is not None:
        return f.ee_pose.position.cpu().numpy().flatten(), f.ee_pose.quaternion.cpu().numpy().flatten()
    return f.ee_position.cpu().numpy().flatten(), f.ee_quaternion.cpu().numpy().flatten()

def run(traj, settle=True):
    """Execute trajectory (arm only — used pre-grasp, when gripper not yet commanded)."""
    for q in traj:
        robot.get_articulation_controller().apply_action(
            ArticulationAction(joint_positions=q.astype(np.float32), joint_indices=ai))
        world.step(render=True)
    if settle:
        fc=traj[-1].astype(np.float32)
        for _ in range(120):
            robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=fc, joint_indices=ai))
            world.step(render=True)
            if np.max(np.abs(robot.get_joint_positions()[ai]-fc))<0.005: break

tq = rotvec_to_quat(TOOL_DOWN_ROTVEC)


# ---------------------------------------------------------------------------
# 7) Free-move planner (orientation held) + straight-Z stitched planner
# ---------------------------------------------------------------------------
def plan_free_move(start_q, target_base, label):
    """Plan a free motion from start_q to target_base position holding the
    tool-down orientation (full pose goal). Returns interpolated waypoints or
    None on failure. Path shape doesn't matter — only that orientation is
    held at the goal (cuRobo keeps the commanded quaternion)."""
    s = JointState.from_position(
        ta.to_device(start_q.astype(np.float32)).view(1, -1),
        joint_names=ARM_JOINT_NAMES)
    g = Pose(position=ta.to_device(target_base.astype(np.float32)).view(1, 3),
             quaternion=ta.to_device(tq.astype(np.float32)).view(1, 4))
    r = mg.plan_single(s, g, MotionGenPlanConfig(max_attempts=5, enable_graph=False))
    if not r.success.item():
        print(f"  [{label}] free move FAILED ({r.status})")
        return None
    return r.get_interpolated_plan().position.cpu().numpy()

def plan_stitched_z(start_q, dz, label, time_dilation=1.0):
    """Plan a Z-only stitched move from start_q by total dz (signed). Returns
    a stitched numpy array of waypoints or None on failure. CASE13_WEIGHT
    holds tool orientation + X + Y and frees Z, so the move is straight down
    or straight up. `time_dilation` < 1 slows the trajectory down."""
    metric = PoseCostMetric(hold_partial_pose=True,
        hold_vec_weight=mg.tensor_args.to_device(np.array(CASE13_WEIGHT, dtype=np.float32)))
    cfg = MotionGenPlanConfig(enable_graph=False, max_attempts=4,
                              enable_finetune_trajopt=False,
                              time_dilation_factor=time_dilation, pose_cost_metric=metric)
    step = dz / N_STEPS
    cur_q = start_q.copy()
    stitched = []
    for i in range(N_STEPS):
        cpos, cquat = fk(cur_q)
        tgt = cpos.copy(); tgt[2] += step
        s = JointState.from_position(ta.to_device(cur_q.astype(np.float32)).view(1, -1),
                                     joint_names=ARM_JOINT_NAMES)
        g = Pose(position=ta.to_device(tgt.astype(np.float32)).view(1, 3),
                 quaternion=ta.to_device(cquat.astype(np.float32)).view(1, 4))
        r = mg.plan_single(s, g, cfg)
        if not r.success.item():
            print(f"  [{label}] step {i+1}/{N_STEPS} FAILED ({r.status})")
            return None
        tr = r.get_interpolated_plan().position.cpu().numpy()
        if stitched: tr = tr[1:]
        stitched.extend(list(tr)); cur_q = tr[-1].copy()
        print(f"  [{label}] planned step {i+1}/{N_STEPS}: target Z={tgt[2]:.4f}")
    return np.array(stitched)

# ---------------------------------------------------------------------------
# 8) Recording state + command helpers
# ---------------------------------------------------------------------------
physics_dt   = float(world.get_physics_dt())
# Recorders are (re)created per object. record_active gates whether rows are
# written; recording is ON only inside the close+hold+open of each grasp.
left_rec = None
right_rec = None
record_active   = False
record_frame    = 0
record_time     = 0.0
record_obj      = ""     # e.g. "cylinder"
record_phase    = ""     # "closing" / "holding" / "opening"

def record_step():
    """Write one row per sensor IF recording is active. No-op otherwise."""
    global record_frame, record_time
    if not record_active or left_rec is None:
        return
    left_rec.record(record_frame, record_time, record_obj, record_phase)
    right_rec.record(record_frame, record_time, record_obj, record_phase)
    record_frame += 1
    record_time  += physics_dt

current_grip = [0.0]   # mutable holder updated by ramp / set
def apply_arm_and_grip(arm_q, grip_val=None):
    if grip_val is not None:
        current_grip[0] = float(grip_val)
    robot.get_articulation_controller().apply_action(
        ArticulationAction(joint_positions=arm_q.astype(np.float32), joint_indices=ai))
    if gi is not None:
        robot.get_articulation_controller().apply_action(
            ArticulationAction(joint_positions=np.array([current_grip[0]], dtype=np.float32),
                               joint_indices=gi))

def run(traj, settle=True):
    """Execute a trajectory, arm only (gripper held at whatever is set).
    Calls record_step() each frame (no-op unless recording is active)."""
    for q in traj:
        apply_arm_and_grip(q)
        world.step(render=True)
        record_step()
    if settle:
        fc = traj[-1].astype(np.float32)
        for _ in range(120):
            apply_arm_and_grip(fc)
            world.step(render=True)
            record_step()
            if np.max(np.abs(robot.get_joint_positions()[ai] - fc)) < 0.005:
                break

def hold_for(arm_q, seconds, label, phase=None):
    global record_phase
    if phase is not None:
        record_phase = phase
    n = int(seconds * 60)
    print(f"[hold] {label}: {seconds:.2f}s ({n} frames)")
    for _ in range(n):
        apply_arm_and_grip(arm_q)
        world.step(render=True)
        record_step()

def log_gripper(label):
    if gi is None:
        return
    actual    = float(robot.get_joint_positions()[gi[0]])
    commanded = float(current_grip[0])
    diff      = actual - commanded
    flag = "  <-- BACK-DRIVEN OPEN" if diff < -0.005 else ""
    print(f"[grip-check] {label}: cmd={commanded:+.4f}  actual={actual:+.4f}  "
          f"diff={diff:+.4f}{flag}")

def ramp_gripper(arm_q, target, label, phase):
    """Ramp the gripper from its current position to `target` over
    GRIPPER_RAMP_FRAMES physics frames. Sets record_phase for the window."""
    global record_phase
    record_phase = phase
    cur_g = float(robot.get_joint_positions()[gi[0]])
    print(f"[grip] {label}: {cur_g:.3f} -> {target:.3f} over "
          f"{GRIPPER_RAMP_FRAMES} frames ({GRIPPER_RAMP_FRAMES/60.0:.2f}s @60Hz cmd).")
    for k in range(GRIPPER_RAMP_FRAMES):
        alpha = (k + 1) / GRIPPER_RAMP_FRAMES
        g_target = cur_g + alpha * (target - cur_g)
        apply_arm_and_grip(arm_q, grip_val=g_target)
        world.step(render=True)
        record_step()

# ---------------------------------------------------------------------------
# 9) Per-object grasp routine (NO lift — object stays on the table)
# ---------------------------------------------------------------------------
arrivals = []   # (name, q_arm) tuples for the TXT report
gripper_timeline = {}   # obj_id -> {close_start, close_end_hold_start, hold_end_open_start, open_end}

def record_arrival(name, q_arm):
    arrivals.append((name, q_arm.copy()))
    fkp, _ = fk(q_arm)
    print(f"[arrive] {name}: FK base pos={fkp}")

def grasp_object(obj_id, up_base, grasp_base, grip_target,
                 up_world, grasp_world, start_q, descend_to_up=False):
    """Full single-object cycle from `start_q`:
        free move to up -> straight down to grasp
        -> [RECORD ON] close -> hold 1s -> open [RECORD OFF]
        -> straight up to up.
    Returns the arm config at the final 'up' pose (start for the next object).
    """
    global left_rec, right_rec, record_active, record_frame, record_time
    global record_obj, record_phase
    print(f"\n========== {obj_id.upper()} ==========")

    # --- reach the UP point ------------------------------------------------
    # descend_to_up=True (object 1): the robot spawns directly above the
    # up-point (same X/Y, just higher Z), so go straight DOWN to it holding
    # orientation + X + Y (CASE13_WEIGHT), exactly like the grasp descent.
    # descend_to_up=False (objects 2/3): free move — the arm has to travel
    # sideways from the previous object's up-point.
    if descend_to_up:
        cur_pos, _ = fk(start_q)
        cur_world  = base_to_world(cur_pos)
        to_up_dz   = float(up_world[2] - cur_world[2])   # signed (negative = down)
        print(f"[{obj_id}] straight descent SPAWN->UP (dz={to_up_dz:+.4f} m, "
              f"orientation+XY held) ...")
        traj_up = plan_stitched_z(start_q, to_up_dz, f"{obj_id}:SPAWN->UP",
                                  time_dilation=DESCENT_TIME_DILATION)
        if traj_up is None:
            print(f"[{obj_id}] SPAWN->UP descent FAILED. Skipping object.")
            return start_q
        run(traj_up)
    else:
        print(f"[{obj_id}] free move to UP {up_world} ...")
        traj_up = plan_free_move(start_q, up_base, f"{obj_id}:to-up")
        if traj_up is None:
            print(f"[{obj_id}] to-up FAILED. Skipping object.")
            return start_q
        run(traj_up)
    q_up = robot.get_joint_positions()[ai].copy()
    record_arrival(f"{obj_id}_UP (approach)", q_up)

    # --- straight descent UP -> GRASP --------------------------------------
    print(f"[{obj_id}] straight descent UP->GRASP ...")
    descent_dz = -float(np.linalg.norm(grasp_world - up_world))
    stitched_dn = plan_stitched_z(q_up, descent_dz, f"{obj_id}:UP->GRASP",
                                  time_dilation=DESCENT_TIME_DILATION)
    if stitched_dn is None:
        print(f"[{obj_id}] descent FAILED. Returning to caller.")
        return q_up
    run(stitched_dn)
    q_grasp = robot.get_joint_positions()[ai].copy()
    record_arrival(f"{obj_id}_GRASP", q_grasp)
    hold_qg = q_grasp.astype(np.float32)

    # --- settle (NOT recorded; recording starts at the close) --------------
    print(f"[{obj_id}] settle {WAIT_GRASP_SECONDS}s before close (not recorded).")
    for _ in range(int(WAIT_GRASP_SECONDS * 60)):
        apply_arm_and_grip(hold_qg)
        world.step(render=True)

    # --- open a fresh recorder pair for THIS object; recording ON ----------
    left_rec, right_rec, left_csv, right_csv = build_recorders(stage, suffix=obj_id)
    print(f"[CSV] path of files is: {left_csv}")
    print(f"[CSV] path of files is: {right_csv}")
    record_frame = 0
    record_time  = 0.0
    record_obj   = obj_id
    record_active = True
    # Tell the TSF_85_Ext extension to START writing CSV rows now (just before
    # the close). Extension keeps the real simulation frame numbers.
    carb.settings.get_settings().set("/exts/TSF_85_Ext/record_active", True)

    try:
        if gi is not None:
            # Frame markers (in RECORDED frames; recording starts at 0 here).
            phase_marks = {}
            phase_marks["close_start"] = record_frame
            # close
            ramp_gripper(hold_qg, grip_target, f"{obj_id} CLOSE", phase="closing")
            log_gripper(f"{obj_id} after close ramp")
            phase_marks["close_end_hold_start"] = record_frame
            # hold
            hold_for(hold_qg, WAIT_HOLD_SECONDS, f"{obj_id} hold closed",
                     phase="holding")
            log_gripper(f"{obj_id} after hold")
            phase_marks["hold_end_open_start"] = record_frame
            # open
            ramp_gripper(hold_qg, GRIPPER_OPEN, f"{obj_id} OPEN", phase="opening")
            log_gripper(f"{obj_id} after open ramp")
            phase_marks["open_end"] = record_frame
            gripper_timeline[obj_id] = dict(phase_marks)
        else:
            print(f"[{obj_id}] gripper joint not found; skipping close/open.")
    finally:
        # recording OFF + close this object's CSV pair before moving up
        record_active = False
        # Tell the TSF_85_Ext extension to STOP writing CSV rows (open done).
        carb.settings.get_settings().set("/exts/TSF_85_Ext/record_active", False)
        print(f"[record] {obj_id} done — closing CSV pair.")
        if left_rec  is not None: left_rec.close();  left_rec  = None
        if right_rec is not None: right_rec.close(); right_rec = None
        record_obj = ""; record_phase = ""

    # --- straight ascent GRASP -> UP (recording already OFF) ---------------
    print(f"[{obj_id}] straight ascent GRASP->UP ...")
    ascent_dz = float(np.linalg.norm(up_world - grasp_world))
    stitched_up = plan_stitched_z(q_grasp, ascent_dz, f"{obj_id}:GRASP->UP",
                                  time_dilation=ASCENT_TIME_DILATION)
    if stitched_up is None:
        print(f"[{obj_id}] ascent FAILED. Returning current pose.")
        return robot.get_joint_positions()[ai].copy()
    run(stitched_up)
    q_up2 = robot.get_joint_positions()[ai].copy()
    record_arrival(f"{obj_id}_UP (retreat)", q_up2)
    return q_up2

# ---------------------------------------------------------------------------
# 10) Run the three objects in sequence
# ---------------------------------------------------------------------------
record_arrival("INITIAL", initial_q)
try:
    q = initial_q.copy()
    # Free move (tool-position IK) to the cylinder up-point — the spawn pose is not
    # directly above the cylinder, so a pure straight-down would miss. Then straight
    # down to grasp, close 1s, open, straight up.
    q = grasp_object("cylinder", cylinder_up_base, cylinder_grasp_base, GRIPPER_CYLINDER,
                     POINT_CYLINDER_UP_WORLD, POINT_CYLINDER_GRASP_WORLD, q,
                     descend_to_up=False)

    # -----------------------------------------------------------------------
    # 11) TXT report — every arrival point, robot + tool pose.
    # -----------------------------------------------------------------------
    def block(name, q_arm):
        wpos, wquat = fk(q_arm)
        world_pos = base_to_world(wpos)
        lines = []
        lines.append(f"--- {name} ---")
        lines.append(f"Joint values (deg): {np.array2string(np.degrees(q_arm), precision=4)}")
        lines.append(f"Joint values (rad): {np.array2string(q_arm, precision=6)}")
        lines.append(f"Tool0 position WORLD (m): {np.array2string(world_pos, precision=6)}")
        lines.append(f"Tool0 position BASE/cuRobo (m): {np.array2string(wpos, precision=6)}")
        lines.append(f"Tool0 orientation quat (wxyz): {np.array2string(wquat, precision=6)}")
        lines.append("")
        return "\n".join(lines)

    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = []
    report.append("UR5e single-object (cylinder) touch pipeline report")
    report.append(f"Generated: {stamp}")
    report.append(f"Robot base in world (m): {np.array2string(ROBOT_WORLD_POS, precision=6)}")
    report.append(f"Joint order: {ARM_JOINT_NAMES}")
    report.append(f"Initial joints (rad): {np.array2string(INITIAL_JOINTS_RAD, precision=6)}")
    report.append("="*60)
    report.append("")
    for name, q_arm in arrivals:
        report.append(block(name, q_arm))
    report.append("="*60)
    report.append("Requested waypoints (world):")
    report.append("  Cylinder up    : " + np.array2string(POINT_CYLINDER_UP_WORLD,    precision=6))
    report.append("  Cylinder grasp : " + np.array2string(POINT_CYLINDER_GRASP_WORLD, precision=6))
    report.append(f"Grasp (close) target (rad): cylinder={GRIPPER_CYLINDER}")

    text = "\n".join(report)
    try:
        with open(OUTPUT_TXT, "w") as f:
            f.write(text)
        print(f"\n[report] Wrote {OUTPUT_TXT}")
    except Exception as e:
        print(f"[report] Could not write {OUTPUT_TXT}: {e}")
    print("\n" + text)

    # -----------------------------------------------------------------------
    # Gripper frame timeline (recorded-frame indices; recording starts at the
    # close, so frame 0 = first close frame). dt = physics_dt seconds/frame.
    # -----------------------------------------------------------------------
    print("\n===== GRIPPER FRAME TIMELINE (recorded frames) =====")
    print(f"physics dt = {physics_dt:.6f} s  ({1.0/physics_dt:.1f} Hz)")
    for oid, m in gripper_timeline.items():
        cs = m.get("close_start")
        ce = m.get("close_end_hold_start")
        he = m.get("hold_end_open_start")
        oe = m.get("open_end")
        def secs(f): return f"{f*physics_dt:.3f}s" if f is not None else "?"
        print(f"\n[{oid}]")
        print(f"  start CLOSING gripper : frame {cs}  (t={secs(cs)})")
        print(f"  CLOSED / arrived      : frame {ce}  (t={secs(ce)})  "
              f"-> close took {ce-cs} frames")
        print(f"  start 1s HOLD wait    : frame {ce}  (t={secs(ce)})")
        print(f"  HOLD done / start OPEN: frame {he}  (t={secs(he)})  "
              f"-> hold took {he-ce} frames")
        print(f"  fully OPEN (rec end)  : frame {oe}  (t={secs(oe)})  "
              f"-> open took {oe-he} frames")
    print("====================================================")

    # -----------------------------------------------------------------------
    # 12) Hold at the final pose. Recording is OFF.
    # -----------------------------------------------------------------------
    print("\n[done] all objects done. Holding final pose. Recording is OFF. "
          "Close window to exit.")
    final_hold = robot.get_joint_positions()[ai].astype(np.float32)
    while simulation_app.is_running():
        apply_arm_and_grip(final_hold)
        world.step(render=True)

finally:
    print("[record] Ensuring CSV files are closed...")
    if left_rec  is not None: left_rec.close()
    if right_rec is not None: right_rec.close()

simulation_app.close()
