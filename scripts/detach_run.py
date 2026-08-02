"""Launch a .bat fully detached from the calling console.

The SI training tasks kept dying with exit 0xC000013A (STATUS_CONTROL_C_EXIT):
an interactive-session Ctrl+C/console-control event was reaching the scheduled
task's process group and killing training. This launcher starts the target
batch file in a NEW PROCESS GROUP with its OWN hidden console, so:

  * CREATE_NEW_PROCESS_GROUP -> Windows disables Ctrl+C delivery to the new
    group unless it installs its own handler, so external Ctrl+C can't kill it;
  * CREATE_NO_WINDOW         -> it gets a private (invisible) console, so it is
    not attached to any other console whose CTRL_CLOSE could propagate, while
    still having a console so `ping`-based waits inside the bat work.

We Popen and exit immediately (no wait): the child is reparented and keeps
running independently of this launcher, the scheduled task, and the Claude
session. Usage:  python detach_run.py <path-to-bat>
"""
import subprocess
import sys

CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000

if len(sys.argv) < 2:
    sys.exit("usage: detach_run.py <bat>")

bat = sys.argv[1]
subprocess.Popen(
    ["cmd.exe", "/c", bat],
    creationflags=CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW,
    close_fds=True,
)
# Do not wait: let the detached process run on its own.
print(f"detached launch of {bat}")
