#!/usr/bin/env python3
"""
Math Equation Calculator
------------------------
Evaluates math expressions safely (no eval()) and solves single-variable
equations numerically. Standard library only.

Run interactively:      python math_calc.py
Or evaluate one line:   python math_calc.py "2^10 + sqrt(16)"
                        python math_calc.py "x^2 - 5x + 6 = 0"

Examples in the prompt:
    >>> 3 + 4 * 2
    >>> sqrt(144) / (2 + 4)
    >>> 2^8                      (^ works as power)
    >>> 5! is NOT supported, use factorial(5)
    >>> let r = 7                (store a variable)
    >>> pi * r^2
    >>> ans * 2                  (ans = last result)
    >>> 2x + 3 = 11              (solves for x)
    >>> x^2 - 5x + 6 = 0         (finds all real roots)
    >>> sin(x) = 0.5
    >>> mode deg                 (trig in degrees; 'mode rad' to go back)
"""

import ast
import math
import operator
import re
import sys

CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf}

BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

HELP = """
Commands:
  <expression>        evaluate, e.g.  (3 + 4) * 2^3 / sqrt(16)
  <equation>          solve, e.g.     x^2 - 5x + 6 = 0
  let name = <expr>   store a variable, e.g.  let r = 7
  vars                show stored variables
  mode deg | rad      set angle mode for trig functions
  range <lo> <hi>     set root-search interval (default -1000 1000)
  help                show this message
  quit                exit

Functions: sqrt cbrt sin cos tan asin acos atan sinh cosh tanh
           log (natural, or log(x, base)) log10 log2 exp abs round
           floor ceil factorial gcd lcm min max radians degrees
Constants: pi e tau inf        Special: ans (last result)
Notes: ^ means power, and 2x or 3(x+1) imply multiplication.
"""


class CalcError(Exception):
    pass


def preprocess(text: str) -> str:
    """Make casual math notation valid Python."""
    text = text.replace("^", "**").replace("×", "*").replace("÷", "/")
    # 2x -> 2*x, 3(x+1) -> 3*(x+1)   (leaves 1e5 and log10( alone)
    text = re.sub(r"(?<![\w.])(\d+\.?\d*)(?![eE][+-]?\d)(?=[a-zA-Z_(])", r"\1*", text)
    # (a)(b) -> (a)*(b),  (a)x -> (a)*x
    text = re.sub(r"\)\s*(?=[\w(])", ")*", text)
    return text


def fmt(value) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if math.isinf(value) or math.isnan(value):
        return str(value)
    if abs(value - round(value)) < 1e-12 and abs(value) < 1e15:
        return str(int(round(value)))
    return f"{value:.10g}"


class Calculator:
    def __init__(self):
        self.vars = {}
        self.ans = 0
        self.degrees = False
        self.lo, self.hi = -1000.0, 1000.0

    # ---------- function table ----------
    def funcs(self):
        d2r = math.radians if self.degrees else (lambda x: x)
        r2d = math.degrees if self.degrees else (lambda x: x)

        def log(x, base=None):
            return math.log(x) if base is None else math.log(x, base)

        def fact(n):
            if n != int(n) or n < 0 or n > 5000:
                raise CalcError("factorial needs an integer between 0 and 5000")
            return math.factorial(int(n))

        return {
            "sqrt": math.sqrt,
            "cbrt": lambda x: math.copysign(abs(x) ** (1 / 3), x),
            "sin": lambda x: math.sin(d2r(x)),
            "cos": lambda x: math.cos(d2r(x)),
            "tan": lambda x: math.tan(d2r(x)),
            "asin": lambda x: r2d(math.asin(x)),
            "acos": lambda x: r2d(math.acos(x)),
            "atan": lambda x: r2d(math.atan(x)),
            "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
            "log": log, "log10": math.log10, "log2": math.log2,
            "exp": math.exp, "abs": abs, "round": round,
            "floor": math.floor, "ceil": math.ceil,
            "factorial": fact, "gcd": math.gcd, "lcm": math.lcm,
            "min": min, "max": max,
            "radians": math.radians, "degrees": math.degrees,
        }

    # ---------- safe evaluator ----------
    def _eval(self, node, local, funcs):
        if isinstance(node, ast.Expression):
            return self._eval(node.body, local, funcs)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return node.value
            raise CalcError("Only numbers are allowed")

        if isinstance(node, ast.BinOp) and type(node.op) in BIN_OPS:
            left = self._eval(node.left, local, funcs)
            right = self._eval(node.right, local, funcs)
            if isinstance(node.op, ast.Pow) and abs(right) > 10000:
                raise CalcError("Exponent too large")
            return BIN_OPS[type(node.op)](left, right)

        if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPS:
            return UNARY_OPS[type(node.op)](self._eval(node.operand, local, funcs))

        if isinstance(node, ast.Name):
            name = node.id
            if name in local:
                return local[name]
            if name == "ans":
                return self.ans
            if name in self.vars:
                return self.vars[name]
            if name in CONSTANTS:
                return CONSTANTS[name]
            raise CalcError(f"Unknown name '{name}'")

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in funcs:
                raise CalcError("Unknown or unsupported function")
            if node.keywords:
                raise CalcError("Keyword arguments are not supported")
            args = [self._eval(a, local, funcs) for a in node.args]
            return funcs[node.func.id](*args)

        raise CalcError("Unsupported syntax")

    def evaluate(self, expr: str, local=None):
        try:
            tree = ast.parse(preprocess(expr.strip()), mode="eval")
        except SyntaxError:
            raise CalcError("Syntax error, check brackets and operators")
        try:
            return self._eval(tree, local or {}, self.funcs())
        except ZeroDivisionError:
            raise CalcError("Division by zero")
        except OverflowError:
            raise CalcError("Number too large")
        except ValueError as e:
            raise CalcError(f"Math domain error ({e})")
        except TypeError:
            raise CalcError("Wrong number or type of arguments")

    # ---------- equation solver ----------
    def solve(self, equation: str):
        if equation.count("=") != 1:
            raise CalcError("An equation needs exactly one '='")
        lhs, rhs = (s.strip() for s in equation.split("="))
        if not lhs or not rhs:
            raise CalcError("Both sides of '=' need an expression")

        combined = f"({preprocess(lhs)}) - ({preprocess(rhs)})"
        try:
            tree = ast.parse(combined, mode="eval")
        except SyntaxError:
            raise CalcError("Syntax error in equation")

        funcs = self.funcs()
        known = set(funcs) | set(CONSTANTS) | set(self.vars) | {"ans"}
        unknowns = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} - known
        if len(unknowns) != 1:
            raise CalcError(
                "Equation must contain exactly one unknown variable "
                f"(found: {', '.join(sorted(unknowns)) or 'none'})"
            )
        var = unknowns.pop()

        def f(x):
            try:
                v = self._eval(tree, {var: x}, funcs)
                return v if isinstance(v, (int, float)) and math.isfinite(v) else None
            except (CalcError, ValueError, ZeroDivisionError, OverflowError, TypeError):
                return None

        return var, self._find_roots(f)

    def _find_roots(self, f, n=40000):
        lo, hi = self.lo, self.hi
        xs = [lo + (hi - lo) * i / n for i in range(n + 1)]
        ys = [f(x) for x in xs]
        roots = []

        def bisect(a, b, fa):
            for _ in range(200):
                m = (a + b) / 2
                fm = f(m)
                if fm is None:
                    return None
                if fm == 0 or (b - a) < 1e-14:
                    return m
                if (fa < 0) == (fm < 0):
                    a, fa = m, fm
                else:
                    b = m
            return (a + b) / 2

        for i in range(n):
            fa, fb = ys[i], ys[i + 1]
            if fa is None or fb is None:
                continue
            if fa == 0:
                roots.append(xs[i])
            elif fa * fb < 0:
                r = bisect(xs[i], xs[i + 1], fa)
                fr = f(r) if r is not None else None
                # reject asymptotes like tan(x) where |f| blows up instead of crossing 0
                if fr is not None and abs(fr) < 1e-6:
                    roots.append(r)
        if ys[n] == 0:
            roots.append(xs[n])

        # roots where the curve only touches zero, e.g. (x-2)^2 = 0
        for i in range(1, n):
            a, b, c = ys[i - 1], ys[i], ys[i + 1]
            if None in (a, b, c) or b == 0:
                continue
            if a * b > 0 and b * c > 0 and abs(b) <= abs(a) and abs(b) <= abs(c):
                l, r = xs[i - 1], xs[i + 1]
                for _ in range(100):
                    m1, m2 = l + (r - l) / 3, r - (r - l) / 3
                    v1, v2 = f(m1), f(m2)
                    if v1 is None or v2 is None:
                        break
                    if abs(v1) < abs(v2):
                        r = m2
                    else:
                        l = m1
                m = (l + r) / 2
                fm = f(m)
                if fm is not None and abs(fm) < 1e-9:
                    roots.append(m)

        roots.sort()
        unique = []
        for r in roots:
            if not unique or abs(r - unique[-1]) > 1e-6:
                unique.append(r)
        return unique

    # ---------- line handler ----------
    def handle(self, line: str):
        line = line.strip()
        if not line:
            return None

        low = line.lower()
        if low == "help":
            return HELP
        if low == "vars":
            if not self.vars:
                return "No variables stored."
            return "\n".join(f"{k} = {fmt(v)}" for k, v in self.vars.items())
        if low.startswith("mode "):
            mode = low.split()[1]
            if mode not in ("deg", "rad"):
                raise CalcError("Use 'mode deg' or 'mode rad'")
            self.degrees = mode == "deg"
            return f"Angle mode: {'degrees' if self.degrees else 'radians'}"
        if low.startswith("range "):
            try:
                lo, hi = (float(p) for p in line.split()[1:3])
            except ValueError:
                raise CalcError("Usage: range <low> <high>")
            if lo >= hi:
                raise CalcError("Low must be less than high")
            self.lo, self.hi = lo, hi
            return f"Root search range: [{fmt(lo)}, {fmt(hi)}]"

        m = re.match(r"let\s+([A-Za-z_]\w*)\s*=\s*(.+)$", line, re.IGNORECASE)
        if m:
            name, expr = m.groups()
            if name in CONSTANTS or name in self.funcs() or name == "ans":
                raise CalcError(f"'{name}' is reserved")
            self.vars[name] = self.evaluate(expr)
            return f"{name} = {fmt(self.vars[name])}"

        if "=" in line:
            var, roots = self.solve(line)
            if not roots:
                return (f"No real solutions found for {var} in "
                        f"[{fmt(self.lo)}, {fmt(self.hi)}]")
            total = len(roots)
            MAX_SHOWN = 10
            if total > MAX_SHOWN:  # periodic functions like sin(x) have endless roots
                roots = sorted(sorted(roots, key=abs)[:MAX_SHOWN])
            lines = [f"{var} = {fmt(round(r, 8))}" for r in roots]
            if total > MAX_SHOWN:
                lines.append(f"... showing the {MAX_SHOWN} closest to 0 "
                             f"({total} roots found in range)")
            return "\n".join(lines)

        result = self.evaluate(line)
        self.ans = result
        return fmt(result)


def repl(calc: Calculator):
    print("Math Equation Calculator  (type 'help' for commands, 'quit' to exit)")
    while True:
        try:
            line = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.strip().lower() in ("quit", "exit", "q"):
            break
        try:
            out = calc.handle(line)
            if out is not None:
                print(out)
        except CalcError as e:
            print(f"Error: {e}")


def main():
    calc = Calculator()
    if len(sys.argv) > 1:
        try:
            print(calc.handle(" ".join(sys.argv[1:])))
        except CalcError as e:
            sys.exit(f"Error: {e}")
    else:
        repl(calc)


if __name__ == "__main__":
    main()
