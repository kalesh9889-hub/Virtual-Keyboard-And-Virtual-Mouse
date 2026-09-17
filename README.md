# Virtual-Keyboard-And-Virtual-Mouse
The AI-Based Virtual Keyboard and Virtual Mouse is a touchless system that uses a webcam, Python, OpenCV, and MediaPipe to detect hand gestures for cursor control, clicking, scrolling, and typing on a virtual keyboard. It improves accessibility, hygiene, and human–computer interaction
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from pynput.keyboard import Controller, Key
import pyautogui
import cv2
import mediapipe as mp
import numpy as np
import sys
import json
import os
from collections import defaultdict

# ---------------- CONFIG ----------------
SMOOTHING = 7.0           # mouse smoothing (higher = smoother, more lag)
CLICK_COOLDOWN = 0.5      # seconds between allowed clicks
DRAG_HOLD = 0.55          # seconds holding pinch to start drag
DISTANCE_SCALE = 0.08     # fraction of frame diagonal used as click threshold
SHOW_FPS = True
CAM_IDX = 0               # camera index
GESTURES_FILE = "gestures.json"
GESTURE_COOLDOWN = 1.0    # seconds between recognized gesture activations
# ----------------------------------------

keyboard_controller = Controller()

# ---------------- Utility: Feature extraction ----------------
def hand_landmarks_to_feature_vec(hand_landmarks, frame_w, frame_h):
    """
    Build a normalized feature vector describing a static hand pose.
    We pick a set of landmarks (tips and some joints), convert to 2D points,
    normalize by subtracting centroid and dividing by hand scale (max dist).
    Returns a 1D numpy array (flattened).
    """
    # choose landmarks indexes that are informative: wrist (0), thumb tip (4), index tip(8),
    # middle tip(12), ring tip(16), pinky tip(20), plus some MCPs for stability (5,9,13,17)
    idxs = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]
    pts = []
    for i in idxs:
        lm = hand_landmarks.landmark[i]
        x = lm.x * frame_w
        y = lm.y * frame_h
        pts.append([x, y])
    pts = np.array(pts)  # shape (N,2)

    # centroid
    centroid = pts.mean(axis=0)
    pts_centered = pts - centroid

    # scale normalization: use max distance from centroid
    dists = np.linalg.norm(pts_centered, axis=1)
    scale = dists.max() if dists.max() > 1e-6 else 1.0
    pts_normalized = pts_centered / scale

    # flatten to a vector
    vec = pts_normalized.flatten()
    # also append a normalized fingertip spread measure (optional)
    spread = np.linalg.norm(pts_normalized[1] - pts_normalized[3])  # thumb-index
    vec = np.append(vec, spread)
    return vec

def compare_feature(a, b):
    """Euclidean distance between vectors (works even if small differences)."""
    a = np.asarray(a); b = np.asarray(b)
    if a.shape != b.shape:
        return np.inf
    return float(np.linalg.norm(a - b))

# ---------------- GestureStore ----------------
class GestureStore:
    def __init__(self, filename=GESTURES_FILE):
        self.filename = filename
        self.templates = defaultdict(list)  # name -> list of vectors
        self.thresholds = {}  # optional per-gesture threshold
        self.load()

    def add_sample(self, name, vec):
        self.templates[name].append(vec)

    def save(self):
        data = {"templates": {}, "thresholds": {}}
        for name, vecs in self.templates.items():
            data["templates"][name] = [v.tolist() for v in vecs]
        for k, v in self.thresholds.items():
            data["thresholds"][k] = float(v)
        with open(self.filename, "w") as f:
            json.dump(data, f)
        print("Gestures saved to", self.filename)

    def load(self):
        if not os.path.exists(self.filename):
            return
        try:
            with open(self.filename, "r") as f:
                data = json.load(f)
            self.templates.clear()
            for name, vecs in data.get("templates", {}).items():
                self.templates[name] = [np.array(v, dtype=float) for v in vecs]
            self.thresholds = {k: float(v) for k, v in data.get("thresholds", {}).items()}
            print("Loaded gestures:", list(self.templates.keys()))
        except Exception as e:
            print("Failed to load gestures:", e)

    def delete(self, name):
        if name in self.templates:
            del self.templates[name]
        if name in self.thresholds:
            del self.thresholds[name]
        self.save()

    def list_gestures(self):
        return list(self.templates.keys())

# ---------------- GUI: VirtualKeyboard + Gesture Trainer ----------------
class VirtualKeyboard(tk.Tk):
    def __init__(self, gesture_store, gesture_config):
        super().__init__()
        self.title("Virtual Keyboard + Virtual Mouse (with Gesture Trainer)")
        self.resizable(False, False)
        self.shift = False
        self.caps = False
        self.button_refs = {}
        self.gesture_store = gesture_store
        self.gesture_config = gesture_config  # dict with training params
        self.recording = False
        self.record_name = ""
        self.record_samples_collected = 0
        self.record_target_samples = 10
        self.recorded_samples = []  # list of feature vectors
        self.gesture_lock = threading.Lock()
        self.create_widgets()

    def create_widgets(self):
        main = ttk.Frame(self, padding=8)
        main.grid(row=0, column=0)

        # Top controls row (keyboard)
        controls = ttk.Frame(main)
        controls.grid(row=0, column=0, sticky="w", pady=(0,6))
        self.mode_var = tk.BooleanVar(value=False)  # False=system, True=local
        ttk.Checkbutton(controls, text="Local input (type into box)", variable=self.mode_var).grid(row=0, column=0, padx=(0,10))
        ttk.Button(controls, text="Clear", command=self.clear_local).grid(row=0, column=1, padx=(0,6))
        ttk.Button(controls, text="Exit", command=self.on_exit).grid(row=0, column=2)

        # Local text area
        self.text = tk.Text(main, width=60, height=6, font=("Segoe UI", 12), wrap="word")
        self.text.grid(row=1, column=0, pady=(0,8))
        self.text.focus_set()

        # Keyboard area (compact; same as earlier)
        rows = [
            ["`", "1","2","3","4","5","6","7","8","9","0","-","=","Backspace"],
            ["Tab","q","w","e","r","t","y","u","i","o","p","[","]","\\"],
            ["Caps","a","s","d","f","g","h","j","k","l",";", "'", "Enter"],
            ["Shift","z","x","c","v","b","n","m",",",".","/","Shift"],
            ["Space"]
        ]

        for r, keys in enumerate(rows, start=2):
            rowframe = ttk.Frame(main)
            rowframe.grid(row=r, column=0, pady=3)
            for k in keys:
                width = 5
                if k in ["Backspace","Enter","Shift","Caps","Tab","Space"]:
                    if k == "Space":
                        width = 40
                    elif k == "Backspace":
                        width = 9
                    elif k == "Enter":
                        width = 9
                    elif k == "Shift":
                        width = 9
                    elif k == "Caps":
                        width = 7
                    elif k == "Tab":
                        width = 7

                label = k if len(k) > 1 else k
                btn = tk.Button(rowframe, text=label.capitalize(), width=width,
                                command=lambda key=k: self.key_press(key))
                btn.pack(side="left", padx=2)
                if k in ("Shift", "Caps"):
                    self.button_refs.setdefault(k, []).append(btn)

        # Status bar
        self.status = ttk.Label(main, text=self.status_text())
        self.status.grid(row=7, column=0, sticky="w", pady=(6,0))

        # Gesture trainer UI (separate frame)
        gframe = ttk.LabelFrame(main, text="Gestures", padding=6)
        gframe.grid(row=0, column=1, rowspan=6, padx=(10,0), sticky="n")

        ttk.Label(gframe, text="Name:").grid(row=0, column=0, sticky="w")
        self.gname_var = tk.StringVar()
        ttk.Entry(gframe, textvariable=self.gname_var, width=20).grid(row=0, column=1, sticky="w")

        ttk.Label(gframe, text="Samples:").grid(row=1, column=0, sticky="w")
        self.sample_count_var = tk.IntVar(value=self.record_target_samples)
        ttk.Spinbox(gframe, from_=3, to=50, textvariable=self.sample_count_var, width=5).grid(row=1, column=1, sticky="w")

        ttk.Label(gframe, text="Match thr:").grid(row=2, column=0, sticky="w")
        self.match_thr_var = tk.DoubleVar(value=0.8)
        ttk.Entry(gframe, textvariable=self.match_thr_var, width=10).grid(row=2, column=1, sticky="w")
        ttk.Label(gframe, text="(lower stricter)").grid(row=2, column=2, sticky="w")

        self.record_btn = ttk.Button(gframe, text="Record", command=self.start_recording)
        self.record_btn.grid(row=3, column=0, columnspan=2, pady=(6,4), sticky="we")
        self.cancel_btn = ttk.Button(gframe, text="Cancel", command=self.cancel_recording, state="disabled")
        self.cancel_btn.grid(row=4, column=0, columnspan=2, sticky="we")

        ttk.Button(gframe, text="Save templates", command=self.save_gestures).grid(row=5, column=0, columnspan=2, pady=(6,4), sticky="we")
        ttk.Button(gframe, text="Reload templates", command=self.reload_gestures).grid(row=6, column=0, columnspan=2, sticky="we")

        ttk.Label(gframe, text="Known gestures:").grid(row=7, column=0, sticky="w", pady=(8,0))
        self.glistbox = tk.Listbox(gframe, height=6, width=25)
        self.glistbox.grid(row=8, column=0, columnspan=2, pady=(0,6))
        self.refresh_gesture_list()

        ttk.Button(gframe, text="Delete selected", command=self.delete_selected).grid(row=9, column=0, columnspan=2, sticky="we")

        # Bind physical keyboard to mirror to local text when Local mode is on
        self.bind_all("<Key>", self._on_physical_key)

    def status_text(self):
        return f"Shift: {'ON' if self.shift else 'OFF'} | CapsLock: {'ON' if self.caps else 'OFF'} | Mode: {'Local' if self.mode_var.get() else 'System'}"

    def clear_local(self):
        self.text.delete("1.0", tk.END)

    def _on_physical_key(self, event):
        if self.mode_var.get() and event.char:
            self.text.insert(tk.END, event.char)

    def key_press(self, key):
        mode_local = self.mode_var.get()
        k = key

        if k == "Shift":
            self.shift = not self.shift
            for b in self.button_refs.get("Shift", []):
                b.config(relief="sunken" if self.shift else "raised")
            self.status.config(text=self.status_text())
            return
        if k == "Caps":
            self.caps = not self.caps
            for b in self.button_refs.get("Caps", []):
                b.config(relief="sunken" if self.caps else "raised")
            self.status.config(text=self.status_text())
            return
        if k == "Tab":
            self._send_text("\t", mode_local)
            return
        if k == "Backspace":
            if mode_local:
                current = self.text.get("1.0", tk.END)[:-1]
                if current:
                    self.text.delete("1.0", tk.END)
                    self.text.insert("1.0", current[:-1] if current[:-1] else "")
            else:
                self._send_system_key(Key.backspace)
            return
        if k == "Enter":
            if mode_local:
                self.text.insert(tk.END, "\n")
            else:
                self._send_system_key(Key.enter)
            return
        if k == "Space":
            self._send_text(" ", mode_local)
            return

        ch = k
        if len(ch) == 1:
            if ch.isalpha():
                should_upper = (self.shift ^ self.caps)
                ch = ch.upper() if should_upper else ch.lower()
            else:
                shifted_map = {
                    "`":"~", "1":"!", "2":"@", "3":"#", "4":"$", "5":"%", "6":"^", "7":"&",
                    "8":"*", "9":"(", "0":")", "-":"_", "=":"+", "[":"{", "]":"}", "\\":"|",
                    ";":":", "'":'"', ",":"<", ".":">", "/":"?"
                }
                if self.shift:
                    ch = shifted_map.get(ch, ch)

            self._send_text(ch, mode_local)

        if self.shift:
            self.shift = False
            for b in self.button_refs.get("Shift", []):
                b.config(relief="sunken" if self.shift else "raised")
            self.status.config(text=self.status_text())

    def _send_text(self, text, local_mode):
        if local_mode:
            self.text.insert(tk.END, text)
            self.text.see(tk.END)
        else:
            # hide briefly so other app can receive keystrokes, then send
            self.withdraw()
            self.after(50, lambda: self._send_system_text_delayed(text))

    def _send_system_text_delayed(self, text):
        for ch in text:
            if ch == "\t":
                keyboard_controller.press(Key.tab); keyboard_controller.release(Key.tab)
            elif ch == "\n":
                keyboard_controller.press(Key.enter); keyboard_controller.release(Key.enter)
            else:
                try:
                    keyboard_controller.press(ch)
                    keyboard_controller.release(ch)
                except Exception:
                    pass
            time.sleep(0.01)
        self.deiconify(); self.lift(); self.focus_force()

    def _send_system_key(self, keyobj):
        self.withdraw()
        self.after(50, lambda: self._press_release_and_restore(keyobj))

    def _press_release_and_restore(self, keyobj):
        try:
            keyboard_controller.press(keyobj); keyboard_controller.release(keyobj)
        except Exception:
            pass
        self.deiconify(); self.lift(); self.focus_force()

    def on_exit(self):
        self.quit()

    # ---------------- Gesture training controls ----------------
    def refresh_gesture_list(self):
        self.glistbox.delete(0, tk.END)
        for g in self.gesture_store.list_gestures():
            thr = self.gesture_store.thresholds.get(g, self.match_thr_var.get())
            self.glistbox.insert(tk.END, f"{g} (thr={thr:.3f})")

    def start_recording(self):
        name = self.gname_var.get().strip()
        if not name:
            messagebox.showerror("Error", "Enter a gesture name first.")
            return
        self.record_target_samples = max(3, int(self.sample_count_var.get()))
        self.record_name = name
        self.recorded_samples = []
        self.record_samples_collected = 0
        self.recording = True
        self.record_btn.config(state="disabled")
        self.cancel_btn.config(state="normal")
        # show info
        messagebox.showinfo("Recording", f"Recording gesture '{name}'. Please hold the pose {self.record_target_samples} times when prompted.")
        # the actual sample collection is driven by the mouse thread which will call `collect_sample(vec)`
        # so here we just toggle flags.

    def cancel_recording(self):
        self.recording = False
        self.record_btn.config(state="normal")
        self.cancel_btn.config(state="disabled")
        self.recorded_samples = []
        self.record_samples_collected = 0
        self.record_name = ""

    def collect_sample(self, vec):
        """Called from the mouse thread when it wants to add a training sample (thread-safe)."""
        with self.gesture_lock:
            if not self.recording:
                return False, 0, self.record_target_samples
            self.recorded_samples.append(vec)
            self.record_samples_collected += 1
            collected = self.record_samples_collected
            target = self.record_target_samples
            # if done, add to store
            if collected >= target:
                # save averaged template(s) — here we store each sample; later we may average
                for v in self.recorded_samples:
                    self.gesture_store.add_sample(self.record_name, v)
                # set threshold for this gesture
                self.gesture_store.thresholds[self.record_name] = float(self.match_thr_var.get())
                self.gesture_store.save()
                # reset recording state
                self.recording = False
                self.record_btn.config(state="normal")
                self.cancel_btn.config(state="disabled")
                self.recorded_samples = []
                self.record_samples_collected = 0
                self.record_name = ""
                # refresh listbox (on main thread)
                self.after(50, self.refresh_gesture_list)
                return True, collected, target
            return False, collected, target

    def save_gestures(self):
        self.gesture_store.save()
        messagebox.showinfo("Saved", f"Saved gestures to {self.gesture_store.filename}")

    def reload_gestures(self):
        self.gesture_store.load()
        self.refresh_gesture_list()
        messagebox.showinfo("Loaded", f"Loaded gestures from {self.gesture_store.filename}")

    def delete_selected(self):
        sel = self.glistbox.curselection()
        if not sel:
            return
        raw = self.glistbox.get(sel[0])
        name = raw.split(" (")[0]
        if messagebox.askyesno("Delete", f"Delete gesture '{name}'?"):
            self.gesture_store.delete(name)
            self.refresh_gesture_list()
            messagebox.showinfo("Deleted", f"Deleted '{name}'")

# ---------------- Mouse Thread w/ gesture recognition ----------------
class VirtualMouseThread(threading.Thread):
    def __init__(self, stop_event, vk_app, gesture_store):
        super().__init__(daemon=True)
        self.stop_event = stop_event
        self.vk_app = vk_app  # VirtualKeyboard instance for callbacks
        self.gesture_store = gesture_store
        self.screen_w, self.screen_h = pyautogui.size()
        self.cap = None

        # mediapipe
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(max_num_hands=1,
                                         min_detection_confidence=0.6,
                                         min_tracking_confidence=0.6)

        # state
        self.prev_x = 0.0
        self.prev_y = 0.0
        self.last_click_time = 0.0
        self.pinch_start_time = None
        self.dragging = False
        self.prev_time = time.time()
        self.last_gesture_time = 0.0
        self.last_gesture_name = None

    def run(self):
        try:
            self.cap = cv2.VideoCapture(CAM_IDX)
            if not self.cap.isOpened():
                print("Cannot open webcam (index {}). Virtual mouse disabled.".format(CAM_IDX))
                return

            while not self.stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    print("Frame read failed, stopping virtual mouse.")
                    break

                frame = cv2.flip(frame, 1)  # mirror
                frame_h, frame_w = frame.shape[:2]
                diag = np.hypot(frame_w, frame_h)
                click_threshold_px = max(3, int(diag * DISTANCE_SCALE))

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb)
                now = time.time()

                recognized_gesture = None

                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]
                    # compute feature vector for gesture recognition + training
                    feature_vec = hand_landmarks_to_feature_vec(hand, frame_w, frame_h)

                    # If app is recording a gesture, collect a sample (thread-safe)
                    recorded_done, collected, target = self.vk_app.collect_sample(feature_vec)
                    if self.vk_app.recording:
                        # draw progress
                        cv2.putText(frame, f"Recording {self.vk_app.record_name}: {collected}/{target}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

                    # Gesture recognition: compare feature to templates
                    best_name, best_dist = None, float("inf")
                    for name, vecs in self.gesture_store.templates.items():
                        for tvec in vecs:
                            d = compare_feature(feature_vec, tvec)
                            if d < best_dist:
                                best_dist = d; best_name = name
                    # determine threshold: per-gesture threshold if present else global var
                    if best_name is not None:
                        thr = self.gesture_store.thresholds.get(best_name, float(self.vk_app.match_thr_var.get()))
                        # note: our features are normalized; thresholds near 0.6-1.2 typical; user may tune
                        if best_dist <= thr:
                            # cooldown for demonstrating events
                            if now - self.last_gesture_time > GESTURE_COOLDOWN:
                                self.last_gesture_time = now
                                self.last_gesture_name = best_name
                                recognized_gesture = best_name

                    # existing mouse control logic (index/middle/thumb)
                    lm_index = hand.landmark[8]
                    lm_middle = hand.landmark[12]
                    lm_thumb = hand.landmark[4]
                    ix = int(lm_index.x * frame_w); iy = int(lm_index.y * frame_h)
                    mx = int(lm_middle.x * frame_w); my = int(lm_middle.y * frame_h)
                    tx = int(lm_thumb.x * frame_w); ty = int(lm_thumb.y * frame_h)

                    screen_x = np.interp(ix, [0, frame_w], [self.screen_w, 0])
                    screen_y = np.interp(iy, [0, frame_h], [0, self.screen_h])

                    # smoothing
                    self.prev_x += (screen_x - self.prev_x) / SMOOTHING
                    self.prev_y += (screen_y - self.prev_y) / SMOOTHING

                    # move mouse
                    try:
                        pyautogui.moveTo(self.prev_x, self.prev_y, _pause=False)
                    except Exception:
                        pass

                    dist_index_middle = int(np.hypot(ix - mx, iy - my))
                    dist_index_thumb = int(np.hypot(ix - tx, iy - ty))

                    # Draw visuals
                    self.mp_draw.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)
                    cv2.circle(frame, (ix, iy), 8, (0, 255, 0), cv2.FILLED)
                    cv2.circle(frame, (mx, my), 6, (255, 0, 0), cv2.FILLED)
                    cv2.circle(frame, (tx, ty), 6, (0, 180, 255), cv2.FILLED)

                    cv2.putText(frame, f"D_mid:{dist_index_middle}", (10, frame_h - 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    cv2.putText(frame, f"Thr:{click_threshold_px}", (10, frame_h - 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

                    # left click using index-middle pinch
                    if dist_index_middle < click_threshold_px:
                        if now - self.last_click_time > CLICK_COOLDOWN and not self.dragging:
                            self.last_click_time = now
                            try:
                                pyautogui.click()
                            except Exception:
                                pass
                            cv2.putText(frame, "LEFT CLICK", (10, frame_h - 110),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)
                        if self.pinch_start_time is None:
                            self.pinch_start_time = now
                        else:
                            held = now - self.pinch_start_time
                            if held > DRAG_HOLD and not self.dragging:
                                self.dragging = True
                                try:
                                    pyautogui.mouseDown()
                                except Exception:
                                    pass
                                cv2.putText(frame, "DRAG START", (10, frame_h - 140),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)
                    else:
                        # reset pinch timer
                        if self.pinch_start_time is not None:
                            self.pinch_start_time = None
                        if self.dragging:
                            self.dragging = False
                            try:
                                pyautogui.mouseUp()
                            except Exception:
                                pass
                            cv2.putText(frame, "DRAG END", (10, frame_h - 140),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 3)

                    # right click using index-thumb pinch
                    if dist_index_thumb < click_threshold_px:
                        if now - self.last_click_time > CLICK_COOLDOWN:
                            self.last_click_time = now
                            try:
                                pyautogui.click(button="right")
                            except Exception:
                                pass
                            cv2.putText(frame, "RIGHT CLICK", (10, frame_h - 170),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 3)

                else:
                    # no hand: reset pinch state & ensure release
                    self.vk_app.recording and None  # no-op
                    if self.dragging:
                        self.dragging = False
                        try:
                            pyautogui.mouseUp()
                        except Exception:
                            pass

                # annotate recognized gesture
                if recognized_gesture:
                    cv2.putText(frame, f"GESTURE: {recognized_gesture}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,200,0), 3)

                # FPS
                if SHOW_FPS:
                    t = time.time()
                    fps = int(1.0 / (t - self.prev_time)) if (t - self.prev_time) > 0 else 0
                    self.prev_time = t
                    cv2.putText(frame, f"FPS: {fps}", (frame_w - 120, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

                cv2.imshow("Virtual Mouse (press 'q' or Esc to close)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    self.stop_event.set()
                    break

        finally:
            if self.cap is not None and self.cap.isOpened():
                try:
                    self.cap.release()
                except Exception:
                    pass
            cv2.destroyAllWindows()

# ---------------- Main ----------------
def main():
    gesture_store = GestureStore()
    stop_event = threading.Event()

    # Create GUI app (pass gesture store)
    app = VirtualKeyboard(gesture_store, gesture_config={})
    mouse_thread = VirtualMouseThread(stop_event, app, gesture_store)
    mouse_thread.start()

    # Ensure GUI closing stops camera thread
    def on_closing():
        stop_event.set()
        mouse_thread.join(timeout=2.0)
        try:
            app.destroy()
        except Exception:
            pass
        cv2.destroyAllWindows()
        try:
            sys.exit(0)
        except SystemExit:
            pass

    app.protocol("WM_DELETE_WINDOW", on_closing)

    try:
        app.mainloop()
    except KeyboardInterrupt:
        on_closing()

if __name__ == "__main__":
    main()
