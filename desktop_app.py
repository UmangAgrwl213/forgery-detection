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

from PIL import Image, ImageTk


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
    return output / f"{image.stem}_forgery_report.png", result.stdout.strip().splitlines()[-1]


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
    parser.add_argument("--weights", type=Path, default=Path("outputs/best_model.pth"))
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
    status = tk.StringVar(value="Choose an image to inspect for likely edited regions.")
    preview = ttk.Label(frame, anchor="center")
    results = queue.Queue()

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
        button.state(["!disabled"])

    def pick():
        selected = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp"), ("All files", "*.*")])
        if not selected:
            return
        image = Path(selected)
        output = Path(__file__).resolve().parent / "outputs" / "desktop" / uuid4().hex
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

    button = ttk.Button(frame, text="Choose image", command=pick)
    button.pack(anchor="w", pady=12)
    ttk.Label(frame, textvariable=status, wraplength=1020).pack(anchor="w")
    preview.pack(fill="both", expand=True)
    root.mainloop()


if __name__ == "__main__":
    main()
