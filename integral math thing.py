import math
import json
import hashlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


# ─────────────────────────────────────────────────────────────────────────────
import sys

# ─────────────────────────────────────────────────────────────────────────────
#  ANALYTICAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def print_progress(name, current, total):
    percent = min(100.0, float(current) / max(1, total) * 100)
    bar_len = 20
    filled = min(bar_len, int(bar_len * current // max(1, total)))
    bar = '█' * filled + '-' * (bar_len - filled)
    sys.stdout.write(f"\r  Searching: [{bar}] {percent:5.1f}% | {name:<15}")
    sys.stdout.flush()

def power_area(a, c, n):
    """Exact area for f(x) = a*|x|^n + c between its two real roots."""
    if a >= 0 or c <= 0:
        return float('inf')
    r = (c / -a) ** (1.0 / n)
    # ∫_{-r}^{r} (a|x|^n + c) dx = 2*(c*r + a*r^(n+1)/(n+1))
    area = 2 * (c * r + a * r ** (n + 1) / (n + 1))
    return area if area > 0 else float('inf')


def power_fit(points, n, precision, name="Power"):
    """Best a, c for f(x) = a*|x|^n + c minimising area above given points."""
    y_max = max(p[1] for p in points)
    best_area = float('inf')
    best = None

    c_start = y_max + precision
    limit = max(30.0, y_max * 5.0)
    total_steps = max(1, int((limit - c_start) / precision))
    step = 0

    c = c_start
    while c < limit:
        step += 1
        if step % max(1, total_steps // 50) == 0:
            print_progress(name, step, total_steps)
            
        a = -float('inf')
        valid = True
        for xi, yi in points:
            if xi == 0:
                if c < yi:
                    valid = False
                    break
                continue
            cand = (yi - c) / abs(xi) ** n
            if cand > a:
                a = cand

        if valid and a < 0 and not math.isinf(a):
            ar = power_area(a, c, n)
            if 0 < ar < best_area:
                best_area = ar
                best = {'a': a, 'c': c, 'area': ar}
            elif ar > best_area * 1.005 and c > 1.5 * y_max:
                break

        c += precision
    
    print_progress(name, total_steps, total_steps)
    return best


def gaussian_fit(points, precision):
    """Best a, c for f(x) = c·exp(-a·x²). Area = c·√(π/a)."""
    best_area = float('inf')
    best = None
    a_start = precision * 10
    limit = 20.0
    total_steps = max(1, int((limit - a_start) / (precision * 10)))
    step = 0
    
    a = a_start
    while a < limit:
        step += 1
        if step % max(1, total_steps // 50) == 0:
            print_progress("Gaussian", step, total_steps)
            
        try:
            c = max(yi * math.exp(a * xi ** 2) for xi, yi in points)
        except OverflowError:
            break
        if c > 0:
            ar = c * math.sqrt(math.pi / a)
            if ar < best_area:
                best_area = ar
                best = {'a': a, 'c': c, 'area': ar}
            elif ar > best_area * 1.005 and a > 0.5:
                break
        a += precision * 10
        
    print_progress("Gaussian", total_steps, total_steps)
    return best


def exp_decay_fit(points, precision):
    """Best a, c for f(x) = c·exp(-a·|x|). Area = 2c/a."""
    best_area = float('inf')
    best = None
    a_start = precision * 10
    limit = 20.0
    total_steps = max(1, int((limit - a_start) / (precision * 10)))
    step = 0
    
    a = a_start
    while a < limit:
        step += 1
        if step % max(1, total_steps // 50) == 0:
            print_progress("Exp. Decay", step, total_steps)
            
        try:
            c = max(yi * math.exp(a * abs(xi)) for xi, yi in points)
        except OverflowError:
            break
        if c > 0:
            ar = 2 * c / a
            if ar < best_area:
                best_area = ar
                best = {'a': a, 'c': c, 'area': ar}
            elif ar > best_area * 1.005 and a > 0.5:
                break
        a += precision * 10
        
    print_progress("Exp. Decay", total_steps, total_steps)
    return best


# ─────────────────────────────────────────────────────────────────────────────
#  OPTIMIZER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class MinimalSurfaceOptimizer:
    POWER_FAMILIES = {
        'Linear (tent)': 1,
        'Quadratic':     2,
        'Cubic':         3,
        'Quartic':       4,
        'Sextic':        6,
        'Octic':         8,
    }

    def __init__(self, precision=0.001):
        self.precision = precision
        self._cache = {}

    def _key(self, points, tag):
        raw = json.dumps(sorted(points), sort_keys=True) + tag
        return hashlib.md5(raw.encode()).hexdigest()

    # ── 2-D ──────────────────────────────────────────────────────────────────
    def optimize_2d(self, points, constraints, enabled_families):
        key = self._key(points, '2d' + "".join(constraints) + "".join(enabled_families))
        if key in self._cache:
            return self._cache[key]

        results = []

        # Power families
        for name, n in self.POWER_FAMILIES.items():
            r = power_fit(points, n, self.precision, name=name)
            if r:
                formula = f"f(x) = {r['a']:.6f}·|x|^{n} + {r['c']:.6f}"
                results.append({'family': name, 'formula': formula, **r})

        # Gaussian
        g = gaussian_fit(points, self.precision)
        if g:
            formula = f"f(x) = {g['c']:.6f}·exp(-{g['a']:.6f}·x²)"
            results.append({'family': 'Gaussian', 'formula': formula, **g})

        # Exponential decay
        e = exp_decay_fit(points, self.precision)
        if e:
            formula = f"f(x) = {e['c']:.6f}·exp(-{e['a']:.6f}·|x|)"
            results.append({'family': 'Exp. Decay', 'formula': formula, **e})

        # Filter results based on constraints and enabled families
        valid_results = []
        for r in results:
            if r['family'] not in enabled_families:
                continue
            if verify_constraints(r, points, constraints):
                valid_results.append(r)

        valid_results.sort(key=lambda x: x['area'])
        out = {'best': valid_results[0] if valid_results else None, 'all': valid_results}
        self._cache[key] = out
        return out

    # ── 2-D Piecewise Fallback ───────────────────────────────────────────────
    def optimize_2d_piecewise(self, points, constraints, enabled_families):
        # 1. Find the split point (the point with the highest y)
        max_idx = max(range(len(points)), key=lambda i: points[i][1])
        x_peak, y_peak = points[max_idx]

        # 2. Split points and constraints
        left_points, left_constraints = [], []
        right_points, right_constraints = [], []

        for i, p in enumerate(points):
            if p[0] <= x_peak:
                left_points.append(p)
                left_constraints.append(constraints[i])
            if p[0] >= x_peak:
                right_points.append(p)
                right_constraints.append(constraints[i])

        # 3. Helper to fit a side
        def fit_side(side_points, side_constraints, is_left):
            best_res = None
            for name in enabled_families:
                n = self.POWER_FAMILIES.get(name)
                c = y_peak
                if n is not None:
                    a = -float('inf')
                    for xi, yi in side_points:
                        if xi == x_peak: continue
                        cand = (yi - c) / (abs(xi - x_peak) ** n)
                        if cand > a:
                            a = cand
                    if a == -float('inf'): a = -1.0 # Default if only 1 point
                    
                    if a < 0:
                        res = {'family': name, 'a': a, 'c': c, 'x_offset': x_peak}
                        res['formula'] = f"{name}: f(x) = {a:.6f}·|x - {x_peak:.6f}|^{n} + {c:.6f}"
                        if verify_constraints(res, side_points, side_constraints):
                            res['area'] = power_area(a, c, n) / 2.0
                            if not best_res or res['area'] < best_res['area']:
                                best_res = res

                elif name == 'Gaussian':
                    a = float('inf')
                    for xi, yi in side_points:
                        if xi == x_peak or yi <= 0: continue
                        cand = -math.log(yi / c) / ((xi - x_peak) ** 2)
                        if cand < a:
                            a = cand
                    if a == float('inf'): a = 1.0
                    if a > 0:
                        res = {'family': name, 'a': a, 'c': c, 'x_offset': x_peak}
                        res['formula'] = f"{name}: f(x) = {c:.6f}·exp(-{a:.6f}·(x - {x_peak:.6f})²)"
                        if verify_constraints(res, side_points, side_constraints):
                            res['area'] = (c * math.sqrt(math.pi / a)) / 2.0
                            if not best_res or res['area'] < best_res['area']:
                                best_res = res

                elif name == 'Exp. Decay':
                    a = float('inf')
                    for xi, yi in side_points:
                        if xi == x_peak or yi <= 0: continue
                        cand = -math.log(yi / c) / abs(xi - x_peak)
                        if cand < a:
                            a = cand
                    if a == float('inf'): a = 1.0
                    if a > 0:
                        res = {'family': name, 'a': a, 'c': c, 'x_offset': x_peak}
                        res['formula'] = f"{name}: f(x) = {c:.6f}·exp(-{a:.6f}·|x - {x_peak:.6f}|)"
                        if verify_constraints(res, side_points, side_constraints):
                            res['area'] = (c / a)
                            if not best_res or res['area'] < best_res['area']:
                                best_res = res

            return best_res

        left_best = fit_side(left_points, left_constraints, True)
        right_best = fit_side(right_points, right_constraints, False)

        if left_best and right_best:
            return {
                'is_piecewise': True,
                'x_peak': x_peak,
                'left': left_best,
                'right': right_best,
                'area': left_best['area'] + right_best['area']
            }
        return None

    # ── 3-D paraboloid ───────────────────────────────────────────────────────
    def optimize_3d(self, points):
        key = self._key(points, '3d')
        if key in self._cache:
            return self._cache[key]

        z_max = max(p[2] for p in points)
        best_sa = float('inf')
        best = None

        c_start = z_max + self.precision
        limit = max(10.001, z_max * 2.0)
        total_steps = max(1, int((limit - c_start) / self.precision))
        step = 0

        c = c_start
        while c < limit:
            step += 1
            if step % max(1, total_steps // 50) == 0:
                print_progress("3D Paraboloid", step, total_steps)
                
            a = -float('inf')
            valid = True
            for xi, yi, zi in points:
                rsq = xi ** 2 + yi ** 2
                if rsq == 0:
                    if c < zi:
                        valid = False
                        break
                    continue
                cand = (zi - c) / rsq
                if cand > a:
                    a = cand

            if valid and a < 0:
                term = 1 - 4 * a * c
                if term >= 0:
                    sa = (math.pi / (6 * a ** 2)) * (term ** 1.5 - 1)
                    if sa < best_sa:
                        best_sa = sa
                        best = {'a': a, 'c': c, 'surface_area': sa,
                                'formula': f"z = {a:.6f}·(x²+y²) + {c:.6f}"}
                    elif sa > best_sa and c > 1.5 * z_max:
                        break

            c += self.precision
            
        print_progress("3D Paraboloid", total_steps, total_steps)

        self._cache[key] = best
        return best

    def clear_cache(self):
        self._cache = {}


# ─────────────────────────────────────────────────────────────────────────────
#  CONSTRAINT VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def eval_2d(family, params, x):
    n = MinimalSurfaceOptimizer.POWER_FAMILIES.get(family)
    x_off = params.get('x_offset', 0.0)
    dx = abs(x - x_off)
    if n is not None:
        return params['a'] * (dx ** n) + params['c']
    if family == 'Gaussian':
        return params['c'] * math.exp(-params['a'] * (dx ** 2))
    if family == 'Exp. Decay':
        return params['c'] * math.exp(-params['a'] * dx)
    return None

def verify_constraints(res, points, constraints):
    """Verify that a set of points satisfies their respective constraints for a given function."""
    for i, (xi, yi) in enumerate(points):
        f_val = eval_2d(res['family'], res, xi)
        mode = constraints[i]
        if mode == 'inside':
            if f_val < yi - 1e-5:
                return False
        else: # touch
            if abs(f_val - yi) > 0.1:
                return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
#  USER INPUT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

from fractions import Fraction

def ask_float(prompt):
    """Safely evaluate math expressions like -5/2 or 1.5."""
    while True:
        raw = input(prompt).strip()
        if not raw:
            continue
        try:
            # Handle fractions like -5/2 directly
            if '/' in raw:
                return float(Fraction(raw))
            # Otherwise use safe eval for things like sqrt(2)
            safe_dict = {"math": math, "sqrt": math.sqrt, "pi": math.pi}
            val = eval(raw, {"__builtins__": {}}, safe_dict)
            return float(val)
        except Exception:
            print("  ✗ Enter a valid number or expression (e.g., -5/2, sqrt(2), 3.14).")


def ask_int(prompt, lo=1):
    while True:
        try:
            v = int(input(prompt))
            if v >= lo:
                return v
            print(f"  ✗ Must be ≥ {lo}.")
        except ValueError:
            print("  ✗ Enter a valid integer.")


# ─────────────────────────────────────────────────────────────────────────────
#  VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_2d(points, result):
    print("  📈 Generating plot...")
    plt.figure(figsize=(10, 6))
    
    # Points
    px = [p[0] for p in points]
    py = [p[1] for p in points]
    plt.scatter(px, py, color='#ff4757', s=100, label='Data Points', zorder=5, edgecolor='black')

    # Domain
    margin = (max(px) - min(px)) * 0.2 if len(px) > 1 else 1.0
    x_min, x_max = min(px) - margin, max(px) + margin
    x_vals = np.linspace(x_min, x_max, 500)
    
    # Function values
    y_vals = []
    if result.get('is_piecewise'):
        x_peak = result['x_peak']
        left_res = result['left']
        right_res = result['right']
        for xv in x_vals:
            if xv <= x_peak:
                y_vals.append(eval_2d(left_res['family'], left_res, xv))
            else:
                y_vals.append(eval_2d(right_res['family'], right_res, xv))
        
        plt.axvline(x=x_peak, color='gray', linestyle='--', alpha=0.5, label='Split Boundary')
        plt.plot(x_vals, y_vals, color='#1e90ff', linewidth=3, label='Piecewise Best', zorder=4)
        formula = f"Left: {left_res['formula']}\nRight: {right_res['formula']}"
        plt.title(f"Minimal Surface Optimizer\n{formula}", fontsize=10, fontweight='bold')
    else:
        for xv in x_vals:
            y_vals.append(eval_2d(result['family'], result, xv))
        plt.plot(x_vals, y_vals, color='#1e90ff', linewidth=3, label=f"Best: {result['family']}", zorder=4)
        formula = result['formula']
        plt.title(f"Minimal Surface Optimizer: {formula}", fontsize=14, fontweight='bold')
    
    plt.xlabel("x", fontsize=12)
    plt.ylabel("f(x)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_3d(points, a, c, formula):
    print("  📈 Generating 3D model...")
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Points
    px = [p[0] for p in points]
    py = [p[1] for p in points]
    pz = [p[2] for p in points]
    ax.scatter(px, py, pz, color='red', s=100, label='Data Points', alpha=1.0)

    # Surface
    r_max = max(math.sqrt(p[0]**2 + p[1]**2) for p in points) * 1.2
    u = np.linspace(-r_max, r_max, 100)
    v = np.linspace(-r_max, r_max, 100)
    X, Y = np.meshgrid(u, v)
    Z = a * (X**2 + Y**2) + c

    # Only plot above the "ground" or relevant range
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, linewidth=0, antialiased=True)
    
    ax.set_title(f"Minimal Surface Area Paraboloid\n{formula}", fontsize=14)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.legend()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   MINIMAL AREA / SURFACE AREA — FUNCTION OPTIMIZER  ║")
    print("╚══════════════════════════════════════════════════════╝")
    print("Finds the function family (quadratic, quartic, Gaussian,")
    print("exponential, etc.) using the LEAST area while passing")
    print("above every point you specify.\n")

    # Dimension
    while True:
        dim = input("Space dimension? [2 = 2-D curve | 3 = 3-D surface]: ").strip()
        if dim in ('2', '3'):
            break
        print("  ✗ Enter 2 or 3.")
    dim = int(dim)

    n_pts = ask_int(f"\nHow many {'(x,y)' if dim==2 else '(x,y,z)'} points? ", 1)

    points = []
    constraints = [] # 'touch' or 'inside'
    print("\n  Enter the coordinates:")
    for i in range(n_pts):
        print(f"  Point {i + 1}:")
        while True:
            xi = ask_float("    x: ")
            yi = ask_float("    y: ")
            if dim == 2:
                if any(p[0] == xi for p in points):
                    print("  ✗ Error: A mathematical function cannot have multiple outputs for the same input x.")
                    continue
                points.append((xi, yi))
                mode = input("    Constraint? [i = point is INSIDE (below) | t = curve must TOUCH point]: ").strip().lower()
                constraints.append('touch' if mode == 't' else 'inside')
                break
            else:
                zi = ask_float("    z: ")
                if any(p[0] == xi and p[1] == yi for p in points):
                    print("  ✗ Error: A 3D function cannot have multiple outputs for the same (x,y) coordinates.")
                    continue
                points.append((xi, yi, zi))
                # For 3D we default to 'inside' for now as paraboloid fitting is complex
                constraints.append('inside')
                break

    # --- STYLE SELECTION ---
    print("\n" + "─"*40)
    print("🎨  SHAPE STYLE SELECTION")
    print("─"*40)
    print("  1: Curved (Quadratic, Gaussian, etc.)")
    print("  2: Straight/Angular (Linear tent)")
    print("  3: All Families")
    style_choice = input("Select style [1-3, default 3]: ").strip()
    
    enabled_families = []
    if style_choice == '1':
        enabled_families = ['Quadratic', 'Cubic', 'Quartic', 'Sextic', 'Octic', 'Gaussian', 'Exp. Decay']
    elif style_choice == '2':
        enabled_families = ['Linear (tent)']
    else:
        enabled_families = list(MinimalSurfaceOptimizer.POWER_FAMILIES.keys()) + ['Gaussian', 'Exp. Decay']

    # --- RESUME & VERIFICATION PHASE ---
    while True:
        print("\n" + "─"*40)
        print("📊  POINT SUMMARY / VERIFICATION")
        print("─"*40)
        if dim == 2:
            print(f"  {'#':<3} | {'x':>10} | {'y':>10} | {'Mode':<8}")
            print(f"  {'-'*3}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
            for i, (px, py) in enumerate(points):
                mode = constraints[i]
                print(f"  {i+1:<3} | {px:10.4f} | {py:10.4f} | {mode:<8}")
        else:
            print(f"  {'#':<3} | {'x':>8} | {'y':>8} | {'z':>8}")
            print(f"  {'-'*3}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
            for i, (px, py, pz) in enumerate(points):
                print(f"  {i+1:<3} | {px:8.4f} | {py:8.4f} | {pz:8.4f}")

        ok = input("\nAre these correct? (y/n): ").strip().lower()
        if ok == 'y':
            break
        elif ok == 'n':
            pt_idx = ask_int("Which point number do you want to change? ", 1)
            if pt_idx > len(points):
                print(f"  ✗ Invalid point number. Max is {len(points)}.")
                continue
            
            i = pt_idx - 1
            print(f"\nUpdating Point {pt_idx}:")
            while True:
                xi = ask_float("    new x: ")
                yi = ask_float("    new y: ")
                if dim == 2:
                    if any(p[0] == xi and idx != i for idx, p in enumerate(points)):
                        print("  ✗ Error: A mathematical function cannot have multiple outputs for the same input x.")
                        continue
                    points[i] = (xi, yi)
                    mode = input("    new constraint? [i/t]: ").strip().lower()
                    constraints[i] = 'touch' if mode == 't' else 'inside'
                    break
                else:
                    zi = ask_float("    new z: ")
                    if any(p[0] == xi and p[1] == yi and idx != i for idx, p in enumerate(points)):
                        print("  ✗ Error: A 3D function cannot have multiple outputs for the same (x,y) coordinates.")
                        continue
                    points[i] = (xi, yi, zi)
                    break
        else:
            print("  ✗ Please type 'y' for yes or 'n' to edit.")

    while True:
        raw = input(f"\nPrecision (e.g. 0.001, must be > 0) [default 0.001]: ").strip()
        try:
            precision = float(raw) if raw else 0.001
            if precision > 0:
                break
            print("  ✗ Precision must be greater than zero.")
        except ValueError:
            print("  ✗ Enter a valid number.")

    print(f"\n  Searching with precision={precision} …\n")
    opt = MinimalSurfaceOptimizer(precision=precision)

    # ── 2-D ──────────────────────────────────────────────────────────────────
    if dim == 2:
        result = opt.optimize_2d(points, constraints, enabled_families)

        is_piecewise = False
        if not result['best']:
            print("\n  ⚠ No valid single function found. Attempting Piecewise Fallback...")
            piecewise_result = opt.optimize_2d_piecewise(points, constraints, enabled_families)
            if not piecewise_result:
                print("\n  ✗ Fallback failed. No valid function found. Try different points.")
                return
            result_obj = piecewise_result
            is_piecewise = True
        else:
            print() # new line after progress bar
            result_obj = result['best']

        if not is_piecewise:
            # Research table
            print("┌─────────────────────────────────────────────────────────────────┐")
            print("│              📊  RESEARCH: Function Family Comparison            │")
            print("├─────────────────────┬───────────────────────────────┬───────────┤")
            print(f"│ {'Family':<19} │ {'Formula':<29} │ {'Area':>9} │")
            print("├─────────────────────┼───────────────────────────────┼───────────┤")
            for r in result['all']:
                area_s = f"{r['area']:.6f}" if r['area'] < 1e15 else "∞"
                tag = " ← BEST" if r is result['best'] else ""
                fam  = (r['family'] + tag)[:19]
                form = r['formula'][:29]
                print(f"│ {fam:<19} │ {form:<29} │ {area_s:>9} │")
            print("└─────────────────────┴───────────────────────────────┴───────────┘")

            print(f"\n✅  OPTIMAL FUNCTION: {result_obj['family']}")
            print(f"    {result_obj['formula']}")
            print(f"    Minimum area = {result_obj['area']:.6f}\n")
        else:
            print("\n✅  OPTIMAL PIECEWISE FUNCTION FOUND")
            print(f"    Split Point (Peak): x = {result_obj['x_peak']:.4f}")
            print(f"    Left Area  = {result_obj['left']['area']:.6f}")
            print(f"    Right Area = {result_obj['right']['area']:.6f}")
            print(f"    Total Area = {result_obj['area']:.6f}\n")
            print(f"    {result_obj['left']['formula']}")
            print(f"    {result_obj['right']['formula']}\n")

        print("    Constraint verification (f(xᵢ) ≥ yᵢ):")
        all_ok = True
        for i, (xi, yi) in enumerate(points):
            val = None
            if is_piecewise:
                if xi <= result_obj['x_peak']:
                    val = eval_2d(result_obj['left']['family'], result_obj['left'], xi)
                else:
                    val = eval_2d(result_obj['right']['family'], result_obj['right'], xi)
            else:
                val = eval_2d(result_obj['family'], result_obj, xi)
            
            val_s = f"{val:8.4f}" if val is not None else "   N/A  "
            
            mode = constraints[i]
            if mode == 'inside':
                ok = val is not None and val >= yi - 1e-5
                req = f"≥ {yi:.4f}"
            else:
                ok = val is not None and abs(val - yi) < 0.1
                req = f"≈ {yi:.4f}"
            
            sym = "✓" if ok else "✗"
            if not ok:
                all_ok = False
            print(f"      {sym}  f({xi:8.4f}) = {val_s}  (required {req})")
            
        if not all_ok:
            print("\n  ⚠  Some constraints failed — try a smaller precision value.")

        plot_2d(points, result_obj)


    # ── 3-D ──────────────────────────────────────────────────────────────────
    else:
        result = opt.optimize_3d(points)
        if not result:
            print("  ✗ No valid paraboloid found.")
            return

        print("✅  OPTIMAL 3-D PARABOLOID:")
        print(f"    {result['formula']}")
        print(f"    Minimum surface area = {result['surface_area']:.6f}\n")

        print("    Constraint verification (z(xᵢ,yᵢ) ≥ zᵢ):")
        a, c = result['a'], result['c']
        for xi, yi, zi in points:
            val = a * (xi**2 + yi**2) + c
            ok  = val >= zi - 1e-6
            sym = "✓" if ok else "✗"
            print(f"      {sym}  z({xi},{yi}) = {val:.6f}  (required ≥ {zi})")
        
        plot_3d(points, a, c, result['formula'])

    print()


if __name__ == "__main__":
    main()