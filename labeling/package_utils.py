import sys, subprocess


def pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
