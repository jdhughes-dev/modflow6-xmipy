"""BMI extension objects that call through a library's function tables.

`get_extension` in the library returns a pointer to a C struct of function
pointers, which starts with the table version and the number of functions.
"""

from ctypes import (
    CFUNCTYPE,
    POINTER,
    Structure,
    byref,
    c_char_p,
    c_double,
    c_int,
    c_void_p,
    cast,
    create_string_buffer,
)
from typing import TYPE_CHECKING, Any

import numpy as np
from bmipy import BmiPointers
from numpy.typing import NDArray

from xmipy.errors import InputError, XMIError
from xmipy.utils import cd
from xmipy.xmi import Xmi

if TYPE_CHECKING:
    from xmipy.xmiwrapper import XmiWrapper

# every argument is passed by reference, so all are void pointers
_status = c_int
_ptr = c_void_p


class XmiTable(Structure):
    """Function table for the ``"xmi"`` extension."""

    _fields_ = [
        ("version", c_int),
        ("nfunctions", c_int),
        ("prepare_time_step", CFUNCTYPE(_status, _ptr)),
        ("do_time_step", CFUNCTYPE(_status)),
        ("finalize_time_step", CFUNCTYPE(_status)),
        ("get_subcomponent_count", CFUNCTYPE(_status, _ptr)),
        ("prepare_solve", CFUNCTYPE(_status, _ptr)),
        ("solve", CFUNCTYPE(_status, _ptr, _ptr)),
        ("finalize_solve", CFUNCTYPE(_status, _ptr)),
        ("get_version", CFUNCTYPE(_status, _ptr)),
        ("get_var_address", CFUNCTYPE(_status, _ptr, _ptr, _ptr, _ptr)),
    ]


XMI_TABLE_VERSION = 1


class PointersTable(Structure):
    """Function table for the ``"bmi:pointers"`` extension."""

    _fields_ = [
        ("version", c_int),
        ("nfunctions", c_int),
        ("get_value_ptr", CFUNCTYPE(_status, _ptr, _ptr)),
        ("get_value_ptr_double", CFUNCTYPE(_status, _ptr, _ptr)),
        ("get_value_ptr_int", CFUNCTYPE(_status, _ptr, _ptr)),
        ("get_value_ptr_bool", CFUNCTYPE(_status, _ptr, _ptr)),
        ("get_var_rank", CFUNCTYPE(_status, _ptr, _ptr)),
        ("get_var_shape", CFUNCTYPE(_status, _ptr, _ptr)),
    ]


POINTERS_TABLE_VERSION = 1


def table_from_pointer(table_ptr: c_void_p, table_type: Any, version: int) -> Any:
    """Cast the pointer from get_extension to a table, checking its version."""
    table = cast(table_ptr, POINTER(table_type)).contents
    if table.version != version:
        raise XMIError(
            f"{table_type.__name__} version {table.version}, expected {version}"
        )
    return table


class XmiExtension(Xmi):
    """The ``"xmi"`` extension of a library wrapped by `XmiWrapper`."""

    def __init__(self, wrapper: "XmiWrapper", table: XmiTable):
        self._wrapper = wrapper
        self._table = table

    def _call(self, name: str, *args: Any) -> None:
        self._wrapper._execute_function(getattr(self._table, name), *args, name=name)

    def prepare_time_step(self, dt: float) -> None:
        with cd(self._wrapper.working_directory):
            c_dt = c_double(dt)
            self._call("prepare_time_step", byref(c_dt))

    def do_time_step(self) -> None:
        with cd(self._wrapper.working_directory):
            self._call("do_time_step")

    def finalize_time_step(self) -> None:
        with cd(self._wrapper.working_directory):
            self._call("finalize_time_step")

    def get_subcomponent_count(self) -> int:
        count = c_int(0)
        self._call("get_subcomponent_count", byref(count))
        return count.value

    def prepare_solve(self, component_id: int = 1) -> None:
        cid = c_int(component_id)
        with cd(self._wrapper.working_directory):
            self._call("prepare_solve", byref(cid))

    def solve(self, component_id: int = 1) -> bool:
        cid = c_int(component_id)
        has_converged = c_int(0)
        with cd(self._wrapper.working_directory):
            self._call("solve", byref(cid), byref(has_converged))
        return has_converged.value == 1

    def finalize_solve(self, component_id: int = 1) -> None:
        cid = c_int(component_id)
        with cd(self._wrapper.working_directory):
            self._call("finalize_solve", byref(cid))

    def get_version(self) -> str:
        len_version = self._wrapper.get_constant_int("BMI_LENVERSION")
        version = create_string_buffer(len_version)
        self._call("get_version", byref(version))
        return version.value.decode("ascii")

    def get_var_address(
        self, var_name: str, component_name: str, subcomponent_name: str = ""
    ) -> str:
        len_var_address = self._wrapper.get_constant_int("BMI_LENVARADDRESS")
        var_address = create_string_buffer(len_var_address)
        self._call(
            "get_var_address",
            c_char_p(component_name.upper().encode()),
            c_char_p(subcomponent_name.upper().encode()),
            c_char_p(var_name.upper().encode()),
            byref(var_address),
        )
        return var_address.value.decode()


class PointersExtension(BmiPointers):
    """The ``"bmi:pointers"`` extension of a library wrapped by `XmiWrapper`.

    A pointer to a variable in ``"bmi:pointers:transient"`` must be fetched
    again after any call that changes the model state.
    """

    def __init__(self, wrapper: "XmiWrapper", table: PointersTable):
        self._wrapper = wrapper
        self._table = table

    def _call(self, name: str, *args: Any, **kwargs: Any) -> None:
        self._wrapper._execute_function(
            getattr(self._table, name), *args, name=name, **kwargs
        )

    def get_var_rank(self, index: int) -> int:
        rank = c_int(0)
        self._call("get_var_rank", byref(c_int(index)), byref(rank))
        return rank.value

    def get_var_shape(self, index: int) -> NDArray[np.int32]:
        rank = self.get_var_rank(index)
        array = np.zeros(rank, dtype=np.int32)
        self._call("get_var_shape", byref(c_int(index)), c_void_p(array.ctypes.data))
        return array

    def get_value_ptr(self, index: int) -> NDArray[Any]:
        # first scalars
        rank = self.get_var_rank(index)
        if rank == 0:
            return self.get_value_ptr_scalar(index)

        shape_tuple = tuple(np.trim_zeros(self.get_var_shape(index)))
        return self._pointer(index, len(shape_tuple), shape_tuple)

    def get_value_ptr_scalar(self, index: int) -> NDArray[Any]:
        return self._pointer(index, 1, (1,))

    def _pointer(self, index: int, ndim: int, shape: Any) -> NDArray[Any]:
        var_type = self._wrapper.get_var_type(index)
        var_type_lower = var_type.lower()
        if var_type_lower.startswith("double"):
            dtype: Any = np.float64
        elif var_type_lower.startswith("float"):
            dtype = np.float32
        elif var_type_lower.startswith("int"):
            dtype = np.int32
        else:
            raise InputError(f"Unsupported value type {var_type!r}")
        arraytype = np.ctypeslib.ndpointer(
            dtype=dtype, ndim=ndim, shape=shape, flags="C"
        )
        values = arraytype()
        self._call(
            "get_value_ptr",
            byref(c_int(index)),
            byref(values),
            detail=f"for variable index {index}",
        )
        return values.contents  # type: ignore[no-any-return]
