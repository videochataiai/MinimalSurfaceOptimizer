# Minimal Surface Optimizer 📐

A robust Python-based optimization engine designed to identify mathematical functions that minimize area (2D) or surface area (3D) for specific datasets.

## Features
- **Multi-Family Fitting**: Supports Linear, Quadratic, Cubic, Quartic, Gaussian, and Exponential Decay families.
- **Piecewise Optimization**: Automatically detects if a single function is insufficient and attempts a two-part piecewise solution.
- **Interactive CLI**: Robust input validation, fractional support (e.g., `-5/2`), and point-by-point editing.
- **3D Surface Area**: Optimizes paraboloids for 3D coordinate sets.
- **Visual Feedback**: Real-time progress bars and high-quality plots using Matplotlib.

## Quick Start
1. Run the script: `python "integral math thing.py"`
2. Enter your points (supports fractions like `-5/2`).
3. Choose your shape style.
4. Review the generated optimal function and area calculation.

## Requirements
- Python 3.x
- NumPy
- Matplotlib
