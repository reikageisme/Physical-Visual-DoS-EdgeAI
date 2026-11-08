"""Read-only PDH counters, own query, English names. No services/settings changed."""
import ctypes
import math
import time

NAMES=['Processor Frequency','% Processor Performance','% Performance Limit',
       'Performance Limit Flags','% Processor Time']


class Value(ctypes.Structure):
    _fields_=[('status',ctypes.c_ulong),('value',ctypes.c_double)]


def valid_value(return_code,status,value):
    return value if return_code==0 and status in (0,1) and math.isfinite(value) else None


class Query:
    def __init__(self,cpu=2):
        self.dll=ctypes.WinDLL('pdh')
        d=self.dll
        d.PdhOpenQueryW.argtypes=[ctypes.c_wchar_p,ctypes.c_size_t,ctypes.POINTER(ctypes.c_void_p)]
        d.PdhOpenQueryW.restype=ctypes.c_ulong
        d.PdhAddEnglishCounterW.argtypes=[ctypes.c_void_p,ctypes.c_wchar_p,ctypes.c_size_t,ctypes.POINTER(ctypes.c_void_p)]
        d.PdhAddEnglishCounterW.restype=ctypes.c_ulong
        d.PdhCollectQueryData.argtypes=[ctypes.c_void_p]
        d.PdhCollectQueryData.restype=ctypes.c_ulong
        d.PdhGetFormattedCounterValue.argtypes=[ctypes.c_void_p,ctypes.c_ulong,ctypes.c_void_p,ctypes.POINTER(Value)]
        d.PdhGetFormattedCounterValue.restype=ctypes.c_ulong
        d.PdhCloseQuery.argtypes=[ctypes.c_void_p]
        d.PdhCloseQuery.restype=ctypes.c_ulong
        self.handle=ctypes.c_void_p()
        code=d.PdhOpenQueryW(None,0,ctypes.byref(self.handle))
        if code:
            raise OSError(f'PDH open: {code:#x}')
        self.counters={}
        try:
            for name in NAMES:
                counter=ctypes.c_void_p()
                code=d.PdhAddEnglishCounterW(self.handle,f'\\Processor Information(0,{cpu})\\{name}',0,ctypes.byref(counter))
                if code:
                    raise OSError(f'PDH add {name}: {code:#x}')
                self.counters[name]=counter
        except BaseException:
            self.close()
            raise
        self.previous=None

    def collect(self):
        start=time.perf_counter_ns()
        code=self.dll.PdhCollectQueryData(self.handle)
        result=dict(collect_status=code,interval_start_ns=self.previous,collected_ns=start,counters={})
        self.previous=start
        for name,counter in self.counters.items():
            value=Value()
            # NOCAP100 is essential: processor performance may exceed nominal.
            rc=self.dll.PdhGetFormattedCounterValue(counter,0x200|0x8000,None,ctypes.byref(value))
            result['counters'][name]=dict(return_code=rc,status=value.status,
                                          value=valid_value(rc,value.status,value.value) if code==0 else None)
        result['read_duration_ns']=time.perf_counter_ns()-start
        return result

    def close(self):
        if self.handle:
            self.dll.PdhCloseQuery(self.handle)
            self.handle=None
