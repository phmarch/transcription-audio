RUNNER=runners/batchalign2

uv run --project "$RUNNER" python - <<'PY'
import sys, site, ipykernel
print("python:", sys.executable)
print("site:", site.getsitepackages())
print("ipykernel:", ipykernel.__file__)
PY