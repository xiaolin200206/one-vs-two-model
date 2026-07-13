#!/usr/bin/env python3
"""
export_onnx.py — export the six trained detectors to ONNX for on-device benchmarking.

Accuracy is measured on the .pt weights (one trusted evaluation routine for all
architectures); the system benchmark runs on the .onnx graphs, because ONNX Runtime
is the deployment runtime and is therefore the artefact whose latency, memory traffic
and heat the paper reports. See the README, Problem 3.

The exported input resolution is FIXED at the value the model was trained with --
a 1280 model must not be fed 640 inputs, and vice versa. cache_benchmark.py resizes
its images to --size, so the two must agree.

    python export_onnx.py --root runs/detect
    python export_onnx.py --root runs/detect --only combined_1280
"""
import argparse
import os

CONFIGS = {
    'combined_640': 640, 'combined_1280': 1280,
    'leaf_640': 640,     'leaf_1280': 1280,
    'pest_640': 640,     'pest_1280': 1280,
}


def main():
    from ultralytics import YOLO
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='runs/detect',
                    help='directory containing <name>/weights/best.pt')
    ap.add_argument('--only', choices=list(CONFIGS), help='export a single configuration')
    ap.add_argument('--opset', type=int, default=None)
    a = ap.parse_args()

    todo = {a.only: CONFIGS[a.only]} if a.only else CONFIGS
    for name, imgsz in todo.items():
        best = os.path.join(a.root, name, 'weights', 'best.pt')
        if not os.path.isfile(best):
            print(f"[skip] {name}: {best} not found")
            continue
        print(f"\nexport {name}  imgsz={imgsz}")
        kw = dict(format='onnx', imgsz=imgsz, simplify=True, dynamic=False)
        if a.opset:
            kw['opset'] = a.opset
        out = YOLO(best).export(**kw)
        print(f"  -> {out}")


if __name__ == '__main__':
    main()
