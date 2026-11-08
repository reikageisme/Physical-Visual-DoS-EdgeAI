"""Documented Windows process-local QoS; no global power/priority changes.

https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setprocessinformation
"""
import ctypes

PROCESS_POWER_THROTTLING = 4
EXECUTION_SPEED = 1


class State(ctypes.Structure):
    _fields_ = [('Version', ctypes.c_ulong), ('ControlMask', ctypes.c_ulong),
                ('StateMask', ctypes.c_ulong)]


def high_readback_ok(policy):
    return bool(policy['control_mask'] & EXECUTION_SPEED) and not bool(policy['state_mask'] & EXECUTION_SPEED)


def snapshot():
    dll = ctypes.WinDLL('kernel32', use_last_error=True)
    dll.GetCurrentProcess.restype = ctypes.c_void_p
    handle = dll.GetCurrentProcess()
    state = State(1,0,0)
    fn = dll.GetProcessInformation
    fn.argtypes = [ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_ulong]
    fn.restype = ctypes.c_int
    if not fn(handle,PROCESS_POWER_THROTTLING,ctypes.byref(state),ctypes.sizeof(state)):
        raise ctypes.WinError(ctypes.get_last_error())
    dll.GetPriorityClass.argtypes = [ctypes.c_void_p]
    dll.GetPriorityClass.restype = ctypes.c_ulong
    return dict(version=state.Version,control_mask=state.ControlMask,state_mask=state.StateMask,
                priority_class=dll.GetPriorityClass(handle))


def set_high():
    before = snapshot()
    dll = ctypes.WinDLL('kernel32', use_last_error=True)
    dll.GetCurrentProcess.restype = ctypes.c_void_p
    state = State(1,EXECUTION_SPEED,0)
    fn = dll.SetProcessInformation
    fn.argtypes = [ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_ulong]
    fn.restype = ctypes.c_int
    if not fn(dll.GetCurrentProcess(),PROCESS_POWER_THROTTLING,ctypes.byref(state),ctypes.sizeof(state)):
        raise ctypes.WinError(ctypes.get_last_error())
    after = snapshot()
    if not high_readback_ok(after) or after['priority_class'] != before['priority_class']:
        raise RuntimeError('HighQoS readback or unchanged priority check failed')
    return dict(before=before,after=after)
