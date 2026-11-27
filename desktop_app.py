"""Local desktop viewer for tampering localization reports."""

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from uuid import uuid4

from PIL import Image, ImageGrab, ImageTk


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def prepare_clipboard_image(output, clipboard_data):
    if isinstance(clipboard_data, Image.Image):
        output.mkdir(parents=True, exist_ok=True)
        image_path = output / "pasted_image.png"
        clipboard_data.convert("RGB").save(image_path, format="PNG")
        return image_path
    if isinstance(clipboard_data, list):
        for item in clipboard_data:
            if not isinstance(item, str):
                continue
            image_path = Path(item)
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                return image_path
    return None


def format_tampering_assessment(summary):
    area = summary["forged_area_percent"]
    assessment = summary.get("tampering_assessment", {})
    decision = assessment.get("decision", "uncalibrated")
    if decision == "signal_detected":
        return (f"Tampering assessment: calibrated signal detected · {area:.3f}% highlighted. "
                "Pixel scores are uncalibrated; this is not proof of manipulation.")
    if decision == "no_signal_detected":
        return (f"Tampering assessment: no calibrated signal detected · {area:.3f}% highlighted. "
                "Pixel scores are uncalibrated; this is not proof of authenticity.")
    if decision == "inconclusive":
        return f"Tampering assessment: inconclusive · {area:.3f}% highlighted. Pixel scores are uncalibrated."
    return (f"Tampering assessment: UNSURE (uncalibrated) · {area:.3f}% highlighted. "
            "The highlighted area is not a calibrated confidence score.")


def analyze(image, weights, output, device="auto"):
    env = os.environ.copy()
    env.update(MPLBACKEND="Agg", MPLCONFIGDIR=str(output / ".matplotlib"), NO_ALBUMENTATIONS_UPDATE="1")
    result = subprocess.run(
        [sys.executable, "predict.py", str(image), "--weights", str(weights), "--output-dir", str(output), "--device", device],
        cwd=Path(__file__).resolve().parent, env=env, capture_output=True, text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "Prediction failed")
    report = output / f"{image.stem}_forgery_report.png"
    summary = json.loads(report.with_suffix(".json").read_text(encoding="utf-8"))
    return report, format_tampering_assessment(summary)


def save_app_assessment(report, tampering_weights, ai_assessment):
    path = report.with_suffix(".json")
    summary = json.loads(path.read_bytes())
    summary["tampering_checkpoint"] = str(tampering_weights)
    summary["ai_generation_assessment"] = ai_assessment
    pending = path.with_suffix(".json.tmp")
    pending.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pending.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_weights = Path("outputs/casia-source-v2-20261127-183558/best_model.pth")
    if not default_weights.is_file() or not default_weights.with_name("heldout_eval.json").is_file():
        default_weights = Path("outputs/best_model.pth")
    parser.add_argument("--weights", type=Path, default=default_weights)
    parser.add_argument("--ai-weights", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    weights = args.weights.resolve()
    if sys.platform == "win32":
        runtime = Path(sys.base_prefix) / "tcl"
        for variable, subdir in (("TCL_LIBRARY", "tcl8.6"), ("TK_LIBRARY", "tk8.6")):
            directory = runtime / subdir
            if directory.is_dir():
                os.environ.setdefault(variable, str(directory))
    root = tk.Tk()
    root.title("Image Forensics")
    root.geometry("1100x620")
    frame = ttk.Frame(root, padding=20)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="Image forensics", font=("Segoe UI", 22)).pack(anchor="w")
    ttk.Label(frame, text=f"Tampering model: {weights.parent.name}/{weights.name}").pack(anchor="w")
    ai_status = tk.StringVar(value="AI-generated image detection: unavailable" if args.ai_weights is None
                             else f"AI model: {args.ai_weights.parent.name}/{args.ai_weights.name}")
    ttk.Label(frame, textvariable=ai_status).pack(anchor="w")
    status = tk.StringVar(value="Paste an image with Ctrl+V, or choose an image to inspect.")
    preview = ttk.Label(frame, anchor="center")
    results = queue.Queue()
    buttons = []

    def poll():
        try:
            report, message, ai_message = results.get_nowait()
        except queue.Empty:
            root.after(150, poll)
            return
        if report:
            with Image.open(report) as image:
                image.thumbnail((1020, 400))
                preview.image = ImageTk.PhotoImage(image)
            preview.configure(image=preview.image)
            status.set(f"{message}\nReport saved at {report}")
        else:
            status.set(message)
        if ai_message:
            ai_status.set(ai_message)
        for button in buttons:
            button.state(["!disabled"])

    def run_analysis(image, output):
        for button in buttons:
            button.state(["disabled"])
        preview.configure(image="")
        status.set(f"Analyzing {image.name}…")
        if args.ai_weights:
            ai_status.set("Analyzing AI-generation signals…")

        def worker():
            try:
                report, message = analyze(image, weights, output, args.device)
            except Exception as error:
                results.put((None, str(error), "AI analysis skipped" if args.ai_weights else None))
                return
            ai_message = None
            ai_assessment = {"decision": "unavailable"}
            if args.ai_weights:
                ai_assessment["checkpoint"] = str(args.ai_weights.resolve())
                try:
                    from ai_detector import classify

                    score, threshold = classify(image, args.ai_weights)
                    ai_assessment.update(model_score=score, threshold=threshold)
                    if threshold > 1:
                        ai_assessment["decision"] = "inconclusive"
                        ai_message = "AI result inconclusive · no usable cutoff met the validation false-alarm target"
                    else:
                        ai_assessment["decision"] = "signal_detected" if score >= threshold else "no_signal_detected"
                        finding = "AI-generation signal detected" if score >= threshold else "No AI-generation signal detected"
                        ai_message = f"{finding} · score {score:.3f} · cutoff {threshold:.3f}"
                except Exception as error:
                    ai_assessment.update(decision="error", error=str(error))
                    ai_message = f"AI analysis failed: {error}"
            try:
                save_app_assessment(report, weights, ai_assessment)
            except (OSError, ValueError) as error:
                message += f"\nCould not save combined results: {error}"
            results.put((report, message, ai_message))

        threading.Thread(target=worker, daemon=True).start()
        root.after(150, poll)

    def pick():
        selected = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp"), ("All files", "*.*")])
        if selected:
            output = Path(__file__).resolve().parent / "outputs" / "desktop" / uuid4().hex
            run_analysis(Path(selected), output)

    def paste():
        if any(button.instate(["disabled"]) for button in buttons):
            return
        try:
            clipboard_data = ImageGrab.grabclipboard()
        except (OSError, tk.TclError) as error:
            status.set(f"Could not read the clipboard: {error}")
            return
        if clipboard_data is None:
            status.set("No image found in the clipboard. Copy an image, or choose an image file.")
            return
        output = Path(__file__).resolve().parent / "outputs" / "desktop" / uuid4().hex
        try:
            image = prepare_clipboard_image(output, clipboard_data)
        except (OSError, ValueError) as error:
            status.set(f"Could not use the clipboard image: {error}")
            return
        if image is None:
            status.set("No supported image found in the clipboard. Copy an image, or choose an image file.")
            return
        run_analysis(image, output)

    button_frame = ttk.Frame(frame)
    button_frame.pack(anchor="w", pady=12)
    choose_button = ttk.Button(button_frame, text="Choose image", command=pick)
    choose_button.pack(side="left", padx=(0, 8))
    paste_button = ttk.Button(button_frame, text="Paste image (Ctrl+V)", command=paste)
    paste_button.pack(side="left")
    buttons.extend((choose_button, paste_button))
    root.bind("<Control-v>", lambda _event: paste())
    root.bind("<Control-V>", lambda _event: paste())
    ttk.Label(frame, textvariable=status, wraplength=1020).pack(anchor="w")
    preview.pack(fill="both", expand=True)
    root.mainloop()


if __name__ == "__main__":
    main()
