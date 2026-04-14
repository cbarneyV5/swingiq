"""
SwingIQ — Hugging Face Spaces Backend
=======================================
Full golf swing analyzer running on HF Spaces free tier.
- Accepts user swing video + optional pro reference video
- Runs MediaPipe pose detection on both
- Produces annotated video with overlay and coaching report
- Returns everything directly to the browser
"""

import os
import uuid
import json
import asyncio
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
STATIC_DIR  = BASE_DIR / "static"
UPLOAD_DIR  = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR  = BASE_DIR / "models"
MODEL_PATH  = str(MODELS_DIR / "pose_landmarker_full.task")
MODEL_URL   = ("https://storage.googleapis.com/mediapipe-models/"
               "pose_landmarker/pose_landmarker_full/float16/latest/"
               "pose_landmarker_full.task")
PRO_PATH    = str(UPLOAD_DIR / "pro_swing.mp4")

for d in [UPLOAD_DIR, RESULTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(title="SwingIQ", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Job store
jobs: dict = {}
executor = ThreadPoolExecutor(max_workers=2)

# ── Constants ──────────────────────────────────────────────────────────────
FONT = cv2.FONT_HERSHEY_SIMPLEX

BG      = (18,  18,  28)
PANEL   = (30,  30,  46)
BORDER  = (50,  50,  70)
WHITE   = (232, 232, 232)
LGRAY   = (160, 160, 168)
GRAY    = (90,  90, 100)
GREEN   = (52,  199,  89)
AMBER   = (255, 179,   0)
ORANGE  = (255, 111,  31)
RED_C   = (220,  50,  47)
PRO_COL = (90,  160, 255)
ACCENT  = (45,  105, 195)

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28)
]
KEY_J = {0,11,12,13,14,15,16,23,24,25,26,27,28}

# ── Utilities ──────────────────────────────────────────────────────────────

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading pose model (~5 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model ready.")

def sc(score):
    if score >= 80: return GREEN
    if score >= 60: return AMBER
    if score >= 40: return ORANGE
    return RED_C

def slabel(score):
    if score >= 80: return "EXCELLENT"
    if score >= 60: return "GOOD"
    if score >= 40: return "FAIR"
    return "NEEDS WORK"

def html_sc(score):
    if score >= 80: return "#34c759"
    if score >= 60: return "#ffb300"
    if score >= 40: return "#ff6f1f"
    return "#dc322f"

# ── Drawing ────────────────────────────────────────────────────────────────

def rrect(f, x, y, w, h, col, alpha=1.0, r=8):
    x,y,w,h = int(x),int(y),int(w),int(h)
    if w < 1 or h < 1: return
    ov = f.copy()
    r  = min(r, w//2, h//2)
    cv2.rectangle(ov,(x+r,y),(x+w-r,y+h),col,-1)
    cv2.rectangle(ov,(x,y+r),(x+w,y+h-r),col,-1)
    for cx,cy in [(x+r,y+r),(x+w-r,y+r),(x+r,y+h-r),(x+w-r,y+h-r)]:
        cv2.circle(ov,(cx,cy),r,col,-1)
    cv2.addWeighted(ov,alpha,f,1-alpha,0,f)

def lbl(f, s, x, y, size=0.48, col=WHITE, bold=False):
    cv2.putText(f,str(s),(int(x),int(y)),FONT,size,col,2 if bold else 1,cv2.LINE_AA)

def draw_ring(f, cx, cy, r, score, title, sub=""):
    col = sc(score)
    cv2.circle(f,(cx,cy),r,(42,42,58),3,cv2.LINE_AA)
    if score > 0:
        cv2.ellipse(f,(cx,cy),(r,r),0,-90,int(-90+3.6*score),col,3,cv2.LINE_AA)
    ns = cv2.getTextSize(str(score),FONT,0.38,1)[0]
    lbl(f,str(score),cx-ns[0]//2,cy+ns[1]//2,0.38,col)
    ts = cv2.getTextSize(title,FONT,0.28,1)[0]
    lbl(f,title,cx-ts[0]//2,cy+r+13,0.28,LGRAY)
    if sub:
        ss = cv2.getTextSize(sub,FONT,0.26,1)[0]
        lbl(f,sub,cx-ss[0]//2,cy+r+24,0.26,GRAY)

def draw_skel(f, pts, col, thick=2, alpha=1.0):
    if not pts: return
    ov = f.copy()
    for a,b in CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(ov,pts[a],pts[b],col,thick)
    for i,(x,y) in enumerate(pts):
        if i in KEY_J:
            cv2.circle(ov,(x,y),5,col,-1)
            cv2.circle(ov,(x,y),5,(0,0,0),1)
    cv2.addWeighted(ov,alpha,f,1-alpha,0,f)

def interp_col(s):
    if s >= 80: return GREEN
    if s >= 60: return AMBER
    if s >= 40: return ORANGE
    return RED_C

def draw_skel_scored(f, pts, jscores, thick=2):
    if not pts: return
    ov = f.copy()
    for a,b in CONNECTIONS:
        if a < len(pts) and b < len(pts):
            avg = (jscores.get(a,70)+jscores.get(b,70))//2
            cv2.line(ov,pts[a],pts[b],interp_col(avg),thick)
    for i,(x,y) in enumerate(pts):
        if i in KEY_J:
            cv2.circle(ov,(x,y),6,interp_col(jscores.get(i,70)),-1)
            cv2.circle(ov,(x,y),6,(0,0,0),1)
    cv2.addWeighted(ov,0.95,f,0.05,0,f)

# ── Math ───────────────────────────────────────────────────────────────────

def ang(a,b,c):
    a,b,c = np.array(a,float),np.array(b,float),np.array(c,float)
    ba,bc = a-b,c-b
    d = np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-7)
    return float(np.degrees(np.arccos(np.clip(d,-1,1))))

def vang(p1,p2):
    return float(np.degrees(np.arctan2(p2[1]-p1[1],p2[0]-p1[0])))

def torso_h(pts):
    if not pts or len(pts)<25: return 1.0
    sh = np.array([(pts[11][0]+pts[12][0])/2,(pts[11][1]+pts[12][1])/2])
    hi = np.array([(pts[23][0]+pts[24][0])/2,(pts[23][1]+pts[24][1])/2])
    return float(np.linalg.norm(sh-hi)) or 1.0

def fit_pro(pro_pts, user_pts):
    if not pro_pts or not user_pts or len(pro_pts)<25 or len(user_pts)<25:
        return None
    scale = torso_h(user_pts)/torso_h(pro_pts)
    ph = np.array([(pro_pts[23][0]+pro_pts[24][0])/2,
                   (pro_pts[23][1]+pro_pts[24][1])/2])
    uh = np.array([(user_pts[23][0]+user_pts[24][0])/2,
                   (user_pts[23][1]+user_pts[24][1])/2])
    return [(int((x-ph[0])*scale+uh[0]),int((y-ph[1])*scale+uh[1]))
            for x,y in pro_pts]

# ── Pose extraction ────────────────────────────────────────────────────────

def extract(path, tag, progress_cb=None):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5)
    frames, pts_list = [], []
    count = 0
    with vision.PoseLandmarker.create_from_options(opts) as lmk:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame.copy())
            ts = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = lmk.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
            if res.pose_landmarks:
                lms = res.pose_landmarks[0]
                pts_list.append([(int(lm.x*fw), int(lm.y*fh)) for lm in lms])
            else:
                pts_list.append(None)
            count += 1
            if progress_cb and count % 10 == 0:
                progress_cb(tag, count, total)
    cap.release()
    return frames, pts_list, fw, fh, fps

def resample(pts_list, n):
    if not pts_list: return [None]*n
    m = len(pts_list)
    return [pts_list[min(int(i/max(1,n-1)*(m-1)),m-1)] for i in range(n)]

# ── Metrics ────────────────────────────────────────────────────────────────

def find_phases(nose_x):
    if len(nose_x) < 6: return None, None
    arr = np.array(nose_x, float)
    sm  = np.convolve(arr, np.ones(5)/5, mode='same')
    pk  = int(np.argmax(np.abs(sm-sm[0])))
    if pk == 0 or pk >= len(sm)-1: return None, None
    af  = np.abs(np.diff(sm[pk:]))
    return (pk, pk+int(np.argmax(af))+1) if len(af) else (pk, None)

def get_phase(fi, n, bp, ip):
    if bp is None: return "swing"
    if fi < max(1, int(bp*0.12)): return "address"
    if fi <= bp:                   return "backswing"
    if ip and fi <= ip:            return "downswing"
    return "follow-through"

def compute_metrics(pts_list, fw, fh):
    nose_x=[]; spines=[]; hip_a=[]; sh_a=[]; wrists=[]; hip_x=[]
    for pts in pts_list:
        if not pts or len(pts)<29: continue
        nose_x.append(pts[0][0])
        spines.append(ang(pts[11],pts[23],pts[25]))
        hip_a.append(abs(vang(pts[23],pts[24])))
        sh_a.append(abs(vang(pts[11],pts[12])))
        wrists.append(max(ang(pts[13],pts[15],pts[23]),
                          ang(pts[14],pts[16],pts[24])))
        hip_x.append((pts[23][0]+pts[24][0])/2)
    if not nose_x: return {}
    bp, ip = find_phases(nose_x)
    tempo  = None
    if bp and ip and ip > bp > 0:
        ds = ip - bp
        if ds > 0: tempo = round(bp/ds, 2)
    hip_rot = round(max(hip_a)-min(hip_a), 1) if len(hip_a)>1 else 0
    sh_rot  = round(max(sh_a)-min(sh_a),   1) if len(sh_a)>1 else 0
    return {
        "spine_avg":   round(sum(spines)/len(spines), 1) if spines else None,
        "spine_range": round(max(spines)-min(spines),  1) if spines else 0,
        "hip_rot":     hip_rot,
        "sh_rot":      sh_rot,
        "x_factor":    round(max(0, sh_rot-hip_rot), 1),
        "head_pct":    round((max(nose_x)-min(nose_x))/fw*100, 1),
        "tempo":       tempo,
        "weight_shift":round((max(hip_x)-min(hip_x))/fw*100, 1) if hip_x else 0,
        "wrist_hinge": round(max(wrists)-min(wrists), 1) if len(wrists)>1 else 0,
        "bp": bp, "ip": ip,
        "spines": spines, "hip_a": hip_a, "sh_a": sh_a,
    }

# ── Coaching intelligence ──────────────────────────────────────────────────

def coach(key, val, pro_val=None):
    if val is None: return None
    pv = pro_val

    if key == "head_pct":
        if val <= 3:
            s,g = 92,"EXCELLENT"
            wim = "Your head is virtually stationary. This is tour-caliber stability."
            wtd = ["Maintain this. Keep your eyes fixed one inch behind the ball through impact.",
                   "On the range, place a tee at nose height and stay below it.",
                   "This is a genuine strength. Focus on other areas."]
        elif val <= 6:
            s,g = 72,"GOOD"
            wim = "Minor head movement. Acceptable and will not significantly hurt ball-striking."
            wtd = ["Fix your gaze on a specific dimple on the back of the ball before every swing.",
                   "Feel your chin stay still while your shoulders turn under it.",
                   "Film yourself from behind and trace your nose position to see the movement clearly."]
        elif val <= 11:
            s,g = 45,"FAIR"
            wim = "Your head is drifting laterally. This shifts the swing arc and causes fat and thin shots."
            wtd = ["Drill: Hold a headcover under your chin. It should stay in the same spot throughout the swing.",
                   "Feel: Imagine a wall one inch to the right of your head. Do not let your head touch it on the backswing.",
                   "Root cause is often a lateral hip sway instead of rotation. Check your hip metric."]
        else:
            s,g = 20,"NEEDS WORK"
            wim = "Significant head movement detected. This is likely your single biggest ball-striking fault."
            wtd = ["Drill: Have someone hold a club shaft vertically beside your head. It should not move.",
                   "Slow-motion practice: Swing at 25% speed until you feel your head staying still, then build up.",
                   "Check your setup. Standing too far from the ball forces the body to sway."]
        cmp = f"Your head moved {val:.1f}% of the frame. The pro moved {pv:.1f}%." if pv else f"Your head moved {val:.1f}% of the frame width. Tour average is under 3%."
        return {"key":"Head Stability","score":s,"grade":g,"reading":f"{val:.1f}% lateral drift",
                "what_it_means":wim,"comparison":cmp,"what_to_do":wtd}

    if key == "hip_rot":
        if val >= 42:
            s,g = 90,"EXCELLENT"
            wim = "Full hip turn. Your lower body is loading correctly, creating a powerful coil."
            wtd = ["On the downswing, fire your lead hip toward the target before your arms start down.",
                   "Feel your trail glute activate on the backswing as you rotate back.",
                   "This is a strength. Protect it by ensuring your weight shifts correctly."]
        elif val >= 33:
            s,g = 70,"GOOD"
            wim = "Solid hip rotation. A small increase would unlock more distance."
            wtd = ["Drill: Flare your trail toe out 20 degrees at address to allow freer hip turn.",
                   "Feel: On the backswing, try to get your trail hip pocket behind your trail heel.",
                   "Hip mobility work off the course will directly translate to more rotation."]
        elif val >= 22:
            s,g = 45,"FAIR"
            wim = "Restricted hip turn. Limited rotation forces your arms to do the work, leading to over-the-top moves and slices."
            wtd = ["Drill: Place a club across your hips. Make backswings until the grip end points at the ball.",
                   "Drill: Lift your trail heel slightly on the backswing to free the hip.",
                   "Feel: Imagine your hips are a door hinge. The door should open 45 degrees on the backswing."]
        else:
            s,g = 22,"NEEDS WORK"
            wim = "Very limited hip rotation. This is likely causing a reverse pivot or steep, choppy swing with significant power loss."
            wtd = ["Drill: Full practice swings with feet together. This forces body rotation and prevents swaying.",
                   "Off-course work: Hip flexor and glute stretches daily for two weeks before re-testing.",
                   "Drill: Make backswings only, pausing at the top and checking your hip position in a mirror."]
        cmp = f"You rotated {val:.0f} degrees. The pro rotated {pv:.0f} degrees." if pv else f"You rotated {val:.0f} degrees. Ideal range is 42 to 50 degrees."
        return {"key":"Hip Rotation","score":s,"grade":g,"reading":f"{val:.0f} degrees of rotation",
                "what_it_means":wim,"comparison":cmp,"what_to_do":wtd}

    if key == "sh_rot":
        if val >= 86:
            s,g = 90,"EXCELLENT"
            wim = "Full shoulder turn. You are maximizing your backswing coil and loading the swing correctly."
            wtd = ["Ensure you are not swaying to achieve this turn. Rotate around a stable spine.",
                   "On the downswing, keep your shoulder turn going through the ball, not stopping at impact.",
                   "This is a strength. Focus on sequencing: hips first, then shoulders, then arms."]
        elif val >= 70:
            s,g = 72,"GOOD"
            wim = "Solid shoulder turn. Getting close to the ideal 90 degrees."
            wtd = ["Feel: Try to get your lead shoulder to point at the ball at the top of your backswing.",
                   "Drill: Cross-arm drill. Hold a club across your chest and make slow-motion backswings.",
                   "Thoracic spine mobility work will help you add the remaining degrees of turn."]
        elif val >= 55:
            s,g = 45,"FAIR"
            wim = "Restricted shoulder turn. You are losing 10 to 20 yards of potential distance."
            wtd = ["Drill: Place a club across your shoulders. The grip end should point at the ball at the top.",
                   "Feel: Your back should face the target at the top of the swing.",
                   "Do thoracic rotation stretches for five minutes before every round."]
        else:
            s,g = 22,"NEEDS WORK"
            wim = "Very short shoulder turn. This forces an arms-only swing with a steep downswing and minimal power."
            wtd = ["Start every practice session with the seated torso rotation stretch for two minutes each side.",
                   "Drill: Make very slow half-swings focused only on turning your back to the target.",
                   "Consider a lesson. A short shoulder turn often has a setup or grip root cause."]
        cmp = f"You turned {val:.0f} degrees. The pro turned {pv:.0f} degrees." if pv else f"You turned {val:.0f} degrees. Tour average is 88 to 95 degrees."
        return {"key":"Shoulder Turn","score":s,"grade":g,"reading":f"{val:.0f} degrees of turn",
                "what_it_means":wim,"comparison":cmp,"what_to_do":wtd}

    if key == "x_factor":
        if val >= 40:
            s,g = 92,"EXCELLENT"
            wim = "Strong X-Factor. The separation between shoulder and hip rotation is generating maximum stored energy. This is the primary biomechanical driver of distance."
            wtd = ["On the downswing, protect this gap by clearing your hips before your shoulders unwind.",
                   "Drill: At the top of your swing, pause and feel the tension in your core. That is X-Factor.",
                   "This is a genuine strength. Keep it."]
        elif val >= 28:
            s,g = 70,"GOOD"
            wim = "Good X-Factor. Solid separation between hips and shoulders. Increasing this gap slightly would meaningfully add distance."
            wtd = ["On the backswing, try to resist with your hips while continuing to turn your shoulders.",
                   "Feel: Your shoulders keep turning after your hips have stopped.",
                   "Drill: Use a resistance band around your hips and make practice backswings."]
        elif val >= 15:
            s,g = 45,"FAIR"
            wim = "Weak X-Factor. Hips and shoulders are turning together, meaning the swing lacks stored rotational energy. This is one of the most common distance killers in amateur golf."
            wtd = ["Core drill: Sit on a chair, hold a club across your chest, and rotate your shoulders while keeping your hips on the seat. Feel the separation.",
                   "Drill: On the backswing, place your lead hand on your lead hip to physically hold it back.",
                   "This is your highest-priority area to improve. Five extra degrees of X-Factor adds approximately eight yards."]
        else:
            s,g = 18,"NEEDS WORK"
            wim = "Very little X-Factor. Hips and shoulders are rotating the same amount. Almost no energy is being stored in the swing."
            wtd = ["Start from scratch. Make very slow backswings focused only on turning shoulders while resisting with hips.",
                   "Core anti-rotation exercises will make this easier to achieve.",
                   "Work on hip rotation and shoulder turn separately before trying to create separation."]
        cmp = f"Your X-Factor is {val:.0f} degrees. The pro's X-Factor is {pv:.0f} degrees." if pv else f"Your X-Factor is {val:.0f} degrees. Tour average is 42 to 50 degrees."
        return {"key":"X-Factor (Power Coil)","score":s,"grade":g,"reading":f"{val:.0f} deg shoulder-hip separation",
                "what_it_means":wim,"comparison":cmp,"what_to_do":wtd}

    if key == "spine_range":
        if val <= 10:
            s,g = 92,"EXCELLENT"
            wim = "Excellent spine angle maintenance. Your forward bend barely changed through the swing."
            wtd = ["Maintain this under pressure. Poor spine angle is the first thing to break down when nervous.",
                   "Keep up your current hip hinge mechanics at address.",
                   "This is a significant strength."]
        elif val <= 22:
            s,g = 70,"GOOD"
            wim = "Mostly stable spine angle. Some change through impact, but within a manageable range."
            wtd = ["Feel: At impact, your chest should still be pointing at the ground, not at the target.",
                   "Drill: Swing in front of a wall standing close. If your backside touches it, your spine angle is rising.",
                   "Strengthening your glutes and hamstrings helps maintain the hip hinge angle."]
        elif val <= 35:
            s,g = 45,"FAIR"
            wim = "You are rising out of your posture through the swing. This causes fat and thin shots."
            wtd = ["Drill: Swing in front of a mirror. Your rear end should stay at the same height from address to impact.",
                   "Feel: Think of staying in the tunnel. Your head and hips should stay at the same height through impact.",
                   "Early extension is often caused by weak glutes. Try hip thrusts and Romanian deadlifts."]
        else:
            s,g = 20,"NEEDS WORK"
            wim = "Significant spine angle change detected. You are standing up dramatically, making consistent contact nearly impossible."
            wtd = ["Drill: Place a towel between your lead armpit and chest. If it falls, you stood up too early.",
                   "Drill: Hit balls off a slight downhill slope. This forces you to stay in your posture.",
                   "Root cause: Often a lack of hip mobility. Work on hip flexor stretches before every session."]
        cmp = f"Your spine angle changed {val:.0f} degrees. The pro's changed {pv:.0f} degrees." if pv else f"Your spine changed {val:.0f} degrees. Tour standard is under 10 degrees."
        return {"key":"Spine Stability","score":s,"grade":g,"reading":f"{val:.0f} deg change through swing",
                "what_it_means":wim,"comparison":cmp,"what_to_do":wtd}

    if key == "tempo":
        if val is None: return None
        if 2.7 <= val <= 3.3:
            s,g = 90,"EXCELLENT"
            wim = "Tour-caliber tempo. Your backswing-to-downswing ratio matches the PGA Tour average of 3:1."
            wtd = ["Maintain this with a pre-shot routine that locks in your rhythm before every swing.",
                   "On the range, use a metronome app at 72 BPM. One beat back, one beat through.",
                   "This is a strength. Protect it by not swinging harder under pressure."]
        elif 2.3 <= val <= 3.7:
            s,g = 68,"GOOD"
            wim = "Tempo is close to ideal. A small adjustment would improve your consistency."
            wtd = ["Count quietly to yourself: 1 and 2 on the backswing, 3 at impact. This creates natural 3:1 rhythm.",
                   "Drill: Swing your lead arm only without a club and feel the natural arc and pause at the top.",
                   "On the course, focus on finishing your backswing fully before starting down."]
        elif val < 2.3:
            s,g = 42,"FAIR"
            wim = "Backswing is rushed. You are starting the downswing before completing the backswing."
            wtd = ["Drill: Pause for one full second at the top of every practice swing.",
                   "Feel: The club should feel weightless at the top before the downswing begins.",
                   "Slow down your takeaway by 30%. Most rushed swings start too fast from the ball."]
        else:
            s,g = 42,"FAIR"
            wim = "Backswing is too slow relative to the downswing. This can cause deceleration through the ball."
            wtd = ["Use a metronome app at 72 BPM to match your swing to a consistent beat.",
                   "Practice with a lighter club to encourage a more natural, free-flowing pace.",
                   "Feel: The downswing should feel like an acceleration, not a lunge."]
        cmp = f"Your tempo is {val:.1f}:1. The pro's tempo is {pv:.1f}:1." if pv else f"Your tempo ratio is {val:.1f}:1. Tour average is 3.0:1."
        return {"key":"Swing Tempo","score":s,"grade":g,"reading":f"{val:.1f}:1 ratio",
                "what_it_means":wim,"comparison":cmp,"what_to_do":wtd}

    if key == "weight_shift":
        if 3 <= val <= 9:
            s,g = 88,"EXCELLENT"
            wim = "Natural, efficient weight shift. Your hips are moving the right amount."
            wtd = ["Ensure this is rotation-driven, not a slide. Your hips should rotate, not just move sideways.",
                   "At finish, 90% of your weight should be on the lead foot. Check this after every swing.",
                   "This is a strength."]
        elif val < 3:
            s,g = 55,"FAIR"
            wim = "Limited weight transfer. Your center of gravity is not moving enough, which reduces power transfer."
            wtd = ["Drill: On the backswing, feel your lead knee move toward the trail knee slightly.",
                   "Feel: Your trail hip should feel loaded and heavy at the top of the backswing.",
                   "Drill: Make slow practice swings while checking your belt buckle faces away from the target at the top."]
        else:
            s,g = 50,"FAIR"
            wim = "Too much lateral hip movement. Your hips are sliding rather than rotating, known as a sway."
            wtd = ["Drill: Place a ball under the outside of your trail foot. It should not roll away on the backswing.",
                   "Feel: Your trail knee should stay in the same position. It bends slightly but does not move outward.",
                   "Think rotate, not slide. Imagine turning inside a barrel."]
        cmp = f"Your hip drift was {val:.1f}% of the frame. The pro's was {pv:.1f}%." if pv else f"Your hip drift was {val:.1f}%. Ideal range is 3 to 9%."
        return {"key":"Weight Shift","score":s,"grade":g,"reading":f"{val:.1f}% lateral hip movement",
                "what_it_means":wim,"comparison":cmp,"what_to_do":wtd}

    if key == "wrist_hinge":
        if val >= 38:
            s,g = 88,"EXCELLENT"
            wim = "Full wrist hinge. You are storing energy correctly in the lever created by your wrists and forearms."
            wtd = ["Ensure this releases fully through impact. Holding the hinge too long causes a push.",
                   "The hinge should happen naturally through gravity and momentum, not a forced flick.",
                   "This is a strength."]
        elif val >= 26:
            s,g = 65,"GOOD"
            wim = "Decent wrist hinge. A small increase would add measurable clubhead speed."
            wtd = ["Let the weight of the clubhead hinge the wrists naturally. Do not resist it.",
                   "Drill: At hip height on the backswing, the shaft should be vertical. Check this in a mirror.",
                   "A slightly stronger grip can help players who tend to hold on too tight."]
        else:
            s,g = 38,"FAIR"
            wim = "Limited wrist hinge. The club is not fully loading at the top. This typically costs 10 to 20 yards."
            wtd = ["Drill: On the backswing, stop when your lead arm is parallel to the ground. The shaft should point straight up.",
                   "Feel: Your trail wrist should be fully cupped and your lead wrist flat at the top.",
                   "Practice hinge-and-hold drills. Swing to waist height and stop. Check the shaft position."]
        cmp = f"Your wrist hinge range was {val:.0f} degrees. The pro's was {pv:.0f} degrees." if pv else f"Your wrist hinge was {val:.0f} degrees. Tour range is typically 40 to 60 degrees."
        return {"key":"Wrist Hinge","score":s,"grade":g,"reading":f"{val:.0f} deg hinge range",
                "what_it_means":wim,"comparison":cmp,"what_to_do":wtd}

    return None

def build_scores(um, pm, has_pro):
    keys    = ["head_pct","hip_rot","sh_rot","x_factor","spine_range","tempo","weight_shift","wrist_hinge"]
    weights = [0.18, 0.14, 0.14, 0.18, 0.16, 0.10, 0.05, 0.05]
    scores  = {}
    for k in keys:
        c = coach(k, um.get(k), pm.get(k) if has_pro else None)
        scores[k] = c["score"] if c else 50
    scores["overall"] = int(sum(scores[k]*w for k,w in zip(keys, weights)))
    return scores

def joint_scores_frame(u_pts, p_pts, phase):
    if not u_pts or not p_pts: return {}
    tol = 12 if phase in ("address","downswing") else 20
    s = {}
    for (a,b,c) in [(11,23,25),(12,24,26),(13,11,23),(14,12,24),(23,25,27),(24,26,28)]:
        if max(a,b,c) < len(u_pts) and max(a,b,c) < len(p_pts):
            diff = abs(ang(u_pts[a],u_pts[b],u_pts[c])-ang(p_pts[a],p_pts[b],p_pts[c]))
            s[b] = max(0, int(100 - diff*(100/(tol*2))))
    for ji in KEY_J:
        if ji not in s: s[ji] = 70
    return s

def estimate_dist(um):
    hf  = np.clip(um.get("hip_rot",0)/45, 0.6, 1.3)
    sf  = np.clip(um.get("sh_rot",0)/90,  0.6, 1.3)
    xf  = np.clip(um.get("x_factor",0)/45,0.85,1.2)
    spf = np.clip((160-(um.get("spine_avg",150) or 150))/20,0.7,1.1)
    t   = um.get("tempo") or 3.0
    tf  = np.clip(1.0-abs(t-3.0)*0.07,0.85,1.1)
    return round(220*hf*sf*xf*spf*tf)

# ── HUD ────────────────────────────────────────────────────────────────────

def draw_hud(f, fw, fh, scores, phase, has_pro):
    pw, ph = 282, 200
    px, py = fw-pw-10, 10
    rrect(f,px,py,pw,ph,PANEL,alpha=0.84,r=10)
    rrect(f,px,py,pw,26,ACCENT,alpha=0.92,r=8)
    header = "VS PRO" if has_pro else "SWING IQ"
    lbl(f,header,px+10,py+17,0.42,WHITE,bold=True)
    phase_col = {"address":AMBER,"backswing":GREEN,"downswing":ORANGE,
                 "follow-through":LGRAY,"swing":WHITE}.get(phase,WHITE)
    ts = cv2.getTextSize(phase.upper(),FONT,0.32,1)[0]
    lbl(f,phase.upper(),px+pw-ts[0]-10,py+17,0.32,phase_col)
    sp = pw//4; r1y = py+62
    rings = [("HEAD",scores["head_pct"],"stability"),
             ("HIPS",scores["hip_rot"],"rotation"),
             ("TURN",scores["sh_rot"],"shoulder"),
             ("SPINE",scores["spine_range"],"posture")]
    for i,(tl,sc2,sub) in enumerate(rings):
        draw_ring(f,px+sp//2+i*sp,r1y,20,sc2,tl,sub)
    dv = r1y+38; cv2.line(f,(px+10,dv),(px+pw-10,dv),BORDER,1)
    r2y = dv+32; sp2 = pw//3
    rings2 = [("X-FACTOR",scores["x_factor"],"power coil"),
              ("TEMPO",scores["tempo"],"rhythm"),
              ("WRIST",scores["wrist_hinge"],"hinge")]
    for i,(tl,sc2,sub) in enumerate(rings2):
        draw_ring(f,px+sp2//2+i*sp2,r2y,18,sc2,tl,sub)
    ov = scores["overall"]; bar_y = r2y+34
    rrect(f,px+8,bar_y,pw-16,26,(42,42,58),r=6)
    fill = max(4,int((pw-16)*ov/100))
    rrect(f,px+8,bar_y,fill,26,sc(ov),r=6)
    lbl(f,f"OVERALL  {ov}/100  {slabel(ov)}",px+14,bar_y+17,0.42,BG,bold=True)
    if has_pro:
        cv2.circle(f,(px+10,py+ph-10),4,GREEN,-1)
        lbl(f,"You",px+17,py+ph-6,0.30,GREEN)
        cv2.circle(f,(px+52,py+ph-10),4,PRO_COL,-1)
        lbl(f,"Pro",px+59,py+ph-6,0.30,PRO_COL)

def make_summary_card(fw, fh, um, pm, scores, dist, has_pro):
    f = np.zeros((fh,fw,3),np.uint8); f[:] = BG
    rrect(f,0,0,fw,60,(26,26,40),r=0)
    lbl(f,"SwingIQ  -  Professional Swing Analysis",18,36,0.82,WHITE,bold=True)
    if has_pro:
        ts = cv2.getTextSize("Scored vs pro reference",FONT,0.34,1)[0]
        lbl(f,"Scored vs pro reference",fw-ts[0]-18,52,0.34,PRO_COL)
    cv2.line(f,(18,64),(fw-18,64),(48,48,68),1)
    ov = scores["overall"]
    cx,cy,r2 = fw//2-90,118,48
    cv2.circle(f,(cx,cy),r2,(40,40,58),4,cv2.LINE_AA)
    if ov > 0:
        cv2.ellipse(f,(cx,cy),(r2,r2),0,-90,int(-90+3.6*ov),sc(ov),4,cv2.LINE_AA)
    ns = cv2.getTextSize(str(ov),FONT,0.82,2)[0]
    lbl(f,str(ov),cx-ns[0]//2,cy+ns[1]//2-2,0.82,sc(ov),bold=True)
    lbl(f,"/ 100",cx+24,cy+ns[1]//2-2,0.30,GRAY)
    lbl(f,"OVERALL SCORE",cx-50,cy+r2+16,0.34,LGRAY)
    lbl(f,slabel(ov),cx-30,cy+r2+32,0.46,sc(ov))
    dist_col = sc(min(100,int(dist/3)))
    rrect(f,fw//2+10,78,fw//2-28,76,(34,34,50),r=10)
    rrect(f,fw//2+10,78,fw//2-28,24,dist_col,r=8)
    lbl(f,"EST CARRY DISTANCE  (DRIVER)",fw//2+18,78+16,0.36,(20,20,20),bold=True)
    ds = f"~{dist} yards"
    ts2 = cv2.getTextSize(ds,FONT,1.05,2)[0]
    lbl(f,ds,fw//2+10+(fw//2-28-ts2[0])//2,78+60,1.05,dist_col,bold=True)
    lbl(f,"Based on rotation mechanics",fw//2+18,78+74,0.30,GRAY)
    cv2.line(f,(18,174),(fw-18,174),(48,48,68),1)
    order = [("x_factor","X-FACTOR  Power Coil"),("head_pct","HEAD  Stability"),
             ("spine_range","SPINE  Posture Maintenance"),("hip_rot","HIPS  Rotation"),
             ("sh_rot","SHOULDERS  Turn"),("tempo","TEMPO  Rhythm"),
             ("weight_shift","WEIGHT  Transfer"),("wrist_hinge","WRISTS  Hinge")]
    ry = 184; bw2 = fw-36; bh = 24
    for k,dname in order:
        c2 = coach(k,um.get(k),pm.get(k) if has_pro else None)
        if not c2: continue
        s2 = c2["score"]; col2 = sc(s2)
        rrect(f,18,ry,bw2,bh,(34,34,50),r=4)
        rrect(f,18,ry,max(4,int(bw2*s2/100)),bh,col2,r=4)
        lbl(f,dname,24,ry+bh-7,0.37,(20,20,20),bold=True)
        g2 = c2["grade"]
        ts3 = cv2.getTextSize(g2,FONT,0.35,1)[0]
        lbl(f,g2,18+bw2-ts3[0]-6,ry+bh-7,0.35,(20,20,20))
        val_line = c2["reading"]
        if has_pro and c2.get("comparison"):
            val_line += "   |   " + c2["comparison"]
        lbl(f,val_line,20,ry+bh+9,0.29,LGRAY)
        ry += bh+18
        if ry > fh-34: break
    rrect(f,0,fh-26,fw,26,(24,24,38),r=0)
    lbl(f,"* Carry distance estimated from biomechanical rotation data. Not radar-measured.",
        18,fh-9,0.31,(66,66,76))
    return f

def make_key_frame_panel(raw, u_pts, p_fit, jsc, phase_label, rows, fw, fh, has_pro):
    f = raw.copy()
    if p_fit:  draw_skel(f,p_fit,PRO_COL,thick=3,alpha=0.52)
    if u_pts:  draw_skel_scored(f,u_pts,jsc,thick=2)
    rrect(f,0,0,fw,48,(12,12,22),alpha=0.90,r=0)
    lbl(f,f"KEY MOMENT:  {phase_label}",12,32,0.72,WHITE,bold=True)
    if has_pro:
        cv2.circle(f,(fw-195,24),5,GREEN,-1)
        lbl(f,"You  (joint color = match quality)",fw-185,28,0.33,GREEN)
        cv2.circle(f,(fw-195,40),5,PRO_COL,-1)
        lbl(f,"Pro reference",fw-185,44,0.33,PRO_COL)
    ph3 = 32*len(rows)+16
    rrect(f,0,fh-ph3,fw,ph3,(12,12,22),alpha=0.88,r=0)
    for i,(k,uv,pv) in enumerate(rows):
        c2 = coach(k,uv,pv)
        if c2:
            col2 = sc(c2["score"])
            line = f"{c2['key'].upper()}:  {c2['grade']}  -  {c2['comparison']}"
        else:
            col2 = GRAY; line = f"{k}: --"
        lbl(f,line,14,fh-ph3+18+i*32,0.44,col2)
    return f

# ── HTML report ────────────────────────────────────────────────────────────

def make_html_report(um, pm, scores, dist, has_pro):
    ov     = scores["overall"]
    ov_col = html_sc(ov)

    metric_order = [
        ("x_factor",    "X-Factor",         "The gap between shoulder and hip rotation — your primary power engine"),
        ("head_pct",    "Head Stability",    "How still your head stayed from address through impact"),
        ("spine_range", "Spine Stability",   "How well you maintained your forward bend through the swing"),
        ("hip_rot",     "Hip Rotation",      "How much your hips turned during the backswing"),
        ("sh_rot",      "Shoulder Turn",     "How far your shoulders rotated at the top of the backswing"),
        ("tempo",       "Swing Tempo",       "Ratio of backswing time to downswing time"),
        ("weight_shift","Weight Shift",      "How efficiently you transferred weight from trail to lead side"),
        ("wrist_hinge", "Wrist Hinge",       "How fully your wrists hinged at the top of the backswing"),
    ]

    cards_html = ""
    for k, name, desc in metric_order:
        c2 = coach(k, um.get(k), pm.get(k) if has_pro else None)
        if not c2: continue
        s2   = c2["score"]
        col  = html_sc(s2)
        drills = "".join(f"<li>{d}</li>" for d in c2["what_to_do"])
        pro_row = (f'<div class="cmp-row"><span class="cmp-icon">vs</span>'
                   f'{c2["comparison"]}</div>') if c2.get("comparison") else ""
        cards_html += f"""
<div class="mcard">
  <div class="mcard-top">
    <div>
      <div class="mcard-name">{name}</div>
      <div class="mcard-desc">{desc}</div>
    </div>
    <div class="mcard-badge" style="background:{col}1a;color:{col};border:1px solid {col}44">{c2["grade"]}</div>
  </div>
  <div class="mbar-track"><div class="mbar-fill" style="width:{s2}%;background:{col}"></div></div>
  <div class="mcard-reading">Your reading: <strong>{c2["reading"]}</strong></div>
  {pro_row}
  <div class="mcard-section-title">What this means</div>
  <div class="mcard-body">{c2["what_it_means"]}</div>
  <div class="mcard-section-title">How to fix it</div>
  <ul class="drill-list">{drills}</ul>
</div>"""

    summary_rows = ""
    for k, name, _ in metric_order:
        c2 = coach(k, um.get(k), pm.get(k) if has_pro else None)
        if not c2: continue
        col = html_sc(c2["score"])
        summary_rows += f"""
<tr>
  <td>{name}</td>
  <td><span style="color:{col};font-weight:600">{c2["grade"]}</span></td>
  <td>{c2["reading"]}</td>
  <td style="color:{col}">{c2["score"]}/100</td>
</tr>"""

    def ds(arr, n=50):
        if not arr: return "[]"
        step = max(1,len(arr)//n)
        return json.dumps([round(arr[i],1) for i in range(0,len(arr),step)])

    u_sp = ds(um.get("spines",[]))
    p_sp = ds(pm.get("spines",[])) if has_pro else "[]"
    u_hi = ds(um.get("hip_a",[]))
    p_hi = ds(pm.get("hip_a",[])) if has_pro else "[]"
    u_sh = ds(um.get("sh_a",[]))
    p_sh = ds(pm.get("sh_a",[])) if has_pro else "[]"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SwingIQ Report</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
:root{{--bg:#0b0b18;--card:#131326;--border:#1c1c38;--text:#dde1ec;--muted:#686880;--faint:#2a2a44}}
body{{font-family:"DM Sans",sans-serif;background:var(--bg);color:var(--text);line-height:1.6;font-size:15px}}
.topbar{{background:#0d0d20;border-bottom:1px solid var(--border);padding:18px 36px;display:flex;align-items:center;gap:12px}}
.logo{{font-family:"DM Serif Display",serif;font-size:22px;color:#fff}}
.logo span{{color:{ov_col}}}
.meta{{font-size:12px;color:var(--muted);margin-left:auto}}
.wrap{{max-width:980px;margin:0 auto;padding:32px 20px}}
.hero{{display:grid;grid-template-columns:auto 1fr auto;gap:28px;align-items:center;background:var(--card);border:1px solid var(--border);border-radius:16px;padding:28px;margin-bottom:24px}}
.score-circle{{width:120px;height:120px;border-radius:50%;display:flex;flex-direction:column;align-items:center;justify-content:center;border:5px solid {ov_col};flex-shrink:0}}
.score-circle .n{{font-size:42px;font-weight:700;color:{ov_col};line-height:1}}
.score-circle .d{{font-size:13px;color:var(--muted);margin-top:1px}}
.hero-mid h2{{font-family:"DM Serif Display",serif;font-size:24px;color:{ov_col};margin-bottom:6px}}
.hero-mid p{{font-size:13px;color:var(--muted);max-width:360px}}
.dist-box{{background:#0a1a0a;border:1px solid #1a3a1a;border-radius:12px;padding:14px 20px;display:inline-block;text-align:center}}
.dist-n{{font-family:"DM Serif Display",serif;font-size:34px;color:#34c759;letter-spacing:-.5px}}
.dist-lbl{{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.6px;margin-top:2px}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:22px;margin-bottom:20px}}
.section-title{{font-size:12px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;margin-bottom:14px;padding-bottom:8px;border-bottom:1px solid var(--border)}}
.summary-table{{width:100%;border-collapse:collapse;font-size:13px}}
.summary-table th{{text-align:left;font-size:11px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.6px;padding:0 0 10px;border-bottom:1px solid var(--border)}}
.summary-table td{{padding:10px 8px 10px 0;border-bottom:1px solid var(--faint);vertical-align:middle}}
.summary-table tr:last-child td{{border-bottom:none}}
.mcards{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.mcard{{background:#0f0f22;border:1px solid var(--border);border-radius:12px;padding:18px}}
.mcard-top{{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;margin-bottom:12px}}
.mcard-name{{font-size:15px;font-weight:600;color:#ececec;margin-bottom:3px}}
.mcard-desc{{font-size:12px;color:var(--muted)}}
.mcard-badge{{font-size:11px;font-weight:600;padding:4px 12px;border-radius:20px;white-space:nowrap;flex-shrink:0}}
.mbar-track{{height:5px;background:#1e1e38;border-radius:3px;margin-bottom:10px}}
.mbar-fill{{height:100%;border-radius:3px}}
.mcard-reading{{font-size:13px;color:var(--muted);margin-bottom:8px}}
.mcard-reading strong{{color:var(--text)}}
.cmp-row{{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--muted);background:#0a0a1e;border-radius:6px;padding:6px 10px;margin-bottom:10px}}
.cmp-icon{{font-size:10px;font-weight:700;background:var(--faint);color:var(--muted);padding:2px 5px;border-radius:4px}}
.mcard-section-title{{font-size:11px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.6px;margin:12px 0 6px}}
.mcard-body{{font-size:13px;color:#b0b4c8;line-height:1.65}}
.drill-list{{font-size:13px;color:#b0b4c8;line-height:1.65;padding-left:18px;margin-top:0}}
.drill-list li{{margin-bottom:6px}}
.chart-grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}}
.legend{{display:flex;gap:14px;margin-bottom:10px}}
.leg{{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--muted)}}
.dot{{width:9px;height:9px;border-radius:50%}}
footer{{text-align:center;font-size:11px;color:#333;padding:28px 20px}}
@media(max-width:700px){{.hero{{grid-template-columns:1fr}}.mcards{{grid-template-columns:1fr}}.chart-grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<div class="topbar">
  <div class="logo">Swing<span>IQ</span></div>
  <div class="meta">{"Scored against your pro reference" if has_pro else "Scored against tour benchmarks"}</div>
</div>
<div class="wrap">
<div class="hero">
  <div class="score-circle"><span class="n">{ov}</span><span class="d">/ 100</span></div>
  <div class="hero-mid">
    <h2>{slabel(ov)}</h2>
    <p>{"Your swing was scored by comparing each metric to the pro reference video you provided." if has_pro else "Your swing was scored against PGA Tour biomechanical benchmarks."}</p>
  </div>
  <div><div class="dist-box"><div class="dist-n">~{dist} yds</div><div class="dist-lbl">Est. carry distance, driver*</div></div></div>
</div>
<div class="card">
  <div class="section-title">Score Summary</div>
  <table class="summary-table">
    <thead><tr><th>Metric</th><th>Grade</th><th>Your Reading</th><th>Score</th></tr></thead>
    <tbody>{summary_rows}</tbody>
  </table>
</div>
<div class="card">
  <div class="section-title">Detailed Coaching Breakdown</div>
  <div class="mcards">{cards_html}</div>
</div>
<div class="card">
  <div class="section-title">Angle Timeline</div>
  {"<div class='legend'><div class='leg'><div class='dot' style='background:#34c759'></div>You</div><div class='leg'><div class='dot' style='background:#5aa0ff'></div>Pro reference</div></div>" if has_pro else "<div class='legend'><div class='leg'><div class='dot' style='background:#34c759'></div>Your swing</div></div>"}
  <div class="chart-grid">
    <div><canvas id="cSpine"></canvas></div>
    <div><canvas id="cHip"></canvas></div>
    <div><canvas id="cShoulder"></canvas></div>
  </div>
</div>
<footer>* Carry distance is estimated from rotation mechanics only and is not radar-measured.</footer>
</div>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script>
function makeChart(id,t,d1,d2,c1,c2){{
  const n=d1.length;const labels=Array.from({{length:n}},(_,i)=>i+1);
  const ds=[{{label:"You",data:d1,borderColor:c1,backgroundColor:c1+"18",borderWidth:2,pointRadius:0,tension:0.4,fill:true}}];
  if(d2&&d2.length)ds.push({{label:"Pro",data:d2,borderColor:c2,backgroundColor:c2+"18",borderWidth:1.5,pointRadius:0,tension:0.4,borderDash:[5,4],fill:true}});
  new Chart(document.getElementById(id),{{type:"line",data:{{labels,datasets:ds}},
    options:{{responsive:true,plugins:{{legend:{{labels:{{color:"#888",font:{{size:11}}}}}},title:{{display:true,text:t,color:"#777",font:{{size:12}}}}}},
    scales:{{x:{{ticks:{{color:"#444",maxTicksLimit:8}},grid:{{color:"#151528"}}}},y:{{ticks:{{color:"#444"}},grid:{{color:"#151528"}}}}}}}}
  }});
}}
makeChart("cSpine","Spine angle across swing",{u_sp},{p_sp},"#34c759","#5aa0ff");
makeChart("cHip","Hip angle across swing",{u_hi},{p_hi},"#34c759","#5aa0ff");
makeChart("cShoulder","Shoulder angle across swing",{u_sh},{p_sh},"#34c759","#5aa0ff");
</script>
</body>
</html>"""

# ── Core processing ────────────────────────────────────────────────────────

def process_swing(job_id: str, user_path: str, pro_path: str):
    def upd(msg, pct):
        jobs[job_id]["message"]  = msg
        jobs[job_id]["progress"] = pct

    try:
        jobs[job_id] = {"status":"processing","progress":0,"message":"Starting...","result":None}
        ensure_model()
        upd("Analyzing your swing...", 10)

        user_frames, user_pts, fw, fh, fps = extract(user_path, "Your swing")
        n = len(user_frames)
        if n == 0:
            raise RuntimeError("No frames found in video.")

        has_pro = os.path.exists(pro_path)
        pro_pts_raw = []; pm = {}
        if has_pro:
            upd("Analyzing pro reference swing...", 25)
            _, pro_pts_raw, pro_fw, pro_fh, _ = extract(pro_path, "Pro swing")
            pm = compute_metrics(pro_pts_raw, pro_fw, pro_fh)
        else:
            upd("No pro reference found. Using tour benchmarks...", 25)

        upd("Computing swing metrics...", 40)
        um = compute_metrics(user_pts, fw, fh)
        if not um:
            raise RuntimeError("No body detected. Please check that your full body is visible and the video is well-lit.")

        scores = build_scores(um, pm, has_pro)
        dist   = estimate_dist(um)

        pro_synced = resample(pro_pts_raw, n)
        pro_fitted = [fit_pro(pro_synced[i], user_pts[i]) for i in range(n)]

        bp = um.get("bp"); ip = um.get("ip")
        addr = next((i for i,p in enumerate(user_pts) if p), 0)
        top  = bp if bp else n//3
        imp  = ip if ip else int(n*0.6)

        def ga(pts):
            if not pts or len(pts)<26: return None,None,None
            return (ang(pts[11],pts[23],pts[25]),
                    abs(vang(pts[23],pts[24])),
                    abs(vang(pts[11],pts[12])))

        upd("Rendering annotated video...", 50)
        out_filename = f"{job_id}.mp4"
        out_path     = str(RESULTS_DIR / out_filename)
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw,fh))

        for fi,(frame,u_pts_f,pf) in enumerate(zip(user_frames,user_pts,pro_fitted)):
            f     = frame.copy()
            phase = get_phase(fi,n,bp,ip)
            jsc   = joint_scores_frame(u_pts_f,pf,phase) if (u_pts_f and pf) else {}
            if pf:      draw_skel(f,pf,PRO_COL,thick=3,alpha=0.50)
            if u_pts_f: draw_skel_scored(f,u_pts_f,jsc,thick=2)
            draw_hud(f,fw,fh,scores,phase,has_pro)
            if bp and fi==bp:
                rrect(f,fw//2-148,8,296,40,(0,80,148),alpha=0.86,r=8)
                lbl(f,"TOP OF BACKSWING",fw//2-128,34,0.68,WHITE,bold=True)
            if ip and fi==ip:
                rrect(f,fw//2-110,8,220,40,(38,38,168),alpha=0.86,r=8)
                lbl(f,"IMPACT ZONE",fw//2-92,34,0.68,WHITE,bold=True)
            out.write(f)
            if fi % 20 == 0:
                upd(f"Rendering frame {fi+1} of {n}...", 50+int((fi/n)*25))

        # Slow motion
        window=22; ims=max(0,imp-window); ime=min(n,imp+window)
        banner = np.zeros((fh,fw,3),np.uint8); banner[:]=BG
        msg2 = "IMPACT ZONE   SLOW MOTION  (4x slower)"
        ts3  = cv2.getTextSize(msg2,FONT,0.80,2)[0]
        lbl(banner,msg2,(fw-ts3[0])//2,fh//2-16,0.80,WHITE,bold=True)
        sub  = "Joint colors show match to pro:  green = matching   red = needs work" if has_pro else "Green = your swing skeleton"
        ts4  = cv2.getTextSize(sub,FONT,0.42,1)[0]
        lbl(banner,sub,(fw-ts4[0])//2,fh//2+22,0.42,LGRAY)
        for _ in range(int(fps*1.8)): out.write(banner)
        for fi in range(ims,ime):
            f      = user_frames[fi].copy()
            pf     = pro_fitted[fi]; u_pts_f = user_pts[fi]
            phase  = get_phase(fi,n,bp,ip)
            jsc    = joint_scores_frame(u_pts_f,pf,phase) if (u_pts_f and pf) else {}
            if pf:      draw_skel(f,pf,PRO_COL,thick=3,alpha=0.50)
            if u_pts_f: draw_skel_scored(f,u_pts_f,jsc,thick=2)
            pct = int((fi-ims)/max(1,(ime-ims)-1)*100)
            rrect(f,fw//2-112,fh-32,224,24,(12,12,22),alpha=0.84,r=5)
            lbl(f,f"Impact zone   {pct}%",fw//2-100,fh-14,0.44,WHITE)
            for _ in range(4): out.write(f)

        # Key frames
        upd("Building key frame panels...", 80)
        for fi, phase_label in [(addr,"ADDRESS"),(top,"TOP OF BACKSWING"),(imp,"IMPACT")]:
            fi      = min(fi,n-1)
            u_pts_f = user_pts[fi]; pf = pro_fitted[fi]
            phase   = get_phase(fi,n,bp,ip)
            jsc     = joint_scores_frame(u_pts_f,pf,phase) if (u_pts_f and pf) else {}
            sp_v,hi_v,sh_v = ga(u_pts_f)
            _,p_hi,p_sh    = ga(pro_synced[fi]) if has_pro else (None,None,None)
            rows = [("spine_range",sp_v,None),("hip_rot",hi_v,p_hi),("sh_rot",sh_v,p_sh)]
            kf   = make_key_frame_panel(user_frames[fi].copy(),u_pts_f,pf,jsc,phase_label,rows,fw,fh,has_pro)
            for _ in range(int(fps*3)): out.write(kf)

        card = make_summary_card(fw,fh,um,pm,scores,dist,has_pro)
        for _ in range(int(fps*8)): out.write(card)
        out.release()

        upd("Generating HTML report...", 92)

        metric_keys = ["x_factor","head_pct","spine_range","hip_rot","sh_rot","tempo","weight_shift","wrist_hinge"]
        coaching = []
        for k in metric_keys:
            c2 = coach(k, um.get(k), pm.get(k) if has_pro else None)
            if c2: coaching.append(c2)

        html_content = make_html_report(um, pm, scores, dist, has_pro)

        result = {
            "overall_score":      scores["overall"],
            "overall_label":      slabel(scores["overall"]),
            "estimated_distance": dist,
            "has_pro":            has_pro,
            "video_filename":     out_filename,
            "scores":             scores,
            "metrics":            {k: um.get(k) for k in metric_keys},
            "coaching":           coaching,
            "html_report":        html_content,
        }

        jobs[job_id]["status"]   = "done"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["message"]  = "Analysis complete."
        jobs[job_id]["result"]   = result

    except Exception as e:
        jobs[job_id]["status"]  = "error"
        jobs[job_id]["message"] = str(e)
        import traceback; traceback.print_exc()
    finally:
        for p in [user_path, pro_path]:
            try:
                if p and os.path.exists(p) and "upload_" in p:
                    os.remove(p)
            except Exception:
                pass

# ── API Routes ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index = STATIC_DIR / "index.html"
    return HTMLResponse(index.read_text(encoding="utf-8"))

@app.post("/api/analyze")
async def analyze(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    pro_video: UploadFile = File(None),
):
    if not video.content_type or not video.content_type.startswith("video/"):
        raise HTTPException(400, "Please upload a valid video file.")

    job_id    = str(uuid.uuid4())[:12]
    user_path = str(UPLOAD_DIR / f"upload_{job_id}_user.mp4")
    pro_path  = str(UPLOAD_DIR / f"upload_{job_id}_pro.mp4")

    contents = await video.read()
    if len(contents) > 300 * 1024 * 1024:
        raise HTTPException(400, "Video file is too large. Please upload a file under 300 MB.")
    with open(user_path, "wb") as f:
        f.write(contents)

    if pro_video and pro_video.filename:
        pro_contents = await pro_video.read()
        with open(pro_path, "wb") as f:
            f.write(pro_contents)
    else:
        pro_path = ""

    jobs[job_id] = {"status":"queued","progress":0,"message":"Upload received. Starting soon...","result":None}
    background_tasks.add_task(process_swing, job_id, user_path, pro_path)

    return JSONResponse({"job_id": job_id})

@app.get("/api/status/{job_id}")
async def status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found.")
    job = jobs[job_id]
    return JSONResponse({
        "status":   job["status"],
        "progress": job["progress"],
        "message":  job["message"],
    })

@app.get("/api/result/{job_id}")
async def result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found.")
    job = jobs[job_id]
    if job["status"] != "done":
        raise HTTPException(400, f"Job is not complete. Status: {job['status']}")
    return JSONResponse(job["result"])

@app.get("/api/video/{filename}")
async def serve_video(filename: str):
    path = RESULTS_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Video not found.")
    return FileResponse(str(path), media_type="video/mp4")

@app.get("/api/health")
async def health():
    return {"status": "ok", "model_ready": os.path.exists(MODEL_PATH)}
