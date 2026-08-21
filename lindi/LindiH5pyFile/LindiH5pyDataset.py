from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlsplit
from urllib.request import url2pathname
import numpy as np
import h5py
import zarr

from .LindiH5pyAttributes import LindiH5pyAttributes
from .LindiH5pyReference import LindiH5pyReference
from ..LindiRemfile.LindiRemfile import LindiRemfile
from ..range_diagnostics import (
    abort_range_read_query,
    complete_range_read_query,
    start_range_read_query,
)

from ..conversion.decode_references import decode_references


if TYPE_CHECKING:
    from .LindiH5pyFile import LindiH5pyFile  # pragma: no cover


class LindiH5pyDataset(h5py.Dataset):
    def __init__(self, _zarr_array: zarr.Array, _file: "LindiH5pyFile"):
        self._zarr_array = _zarr_array
        self._file = _file
        self._readonly = _file.mode not in ['r+']

        # see comment in LindiH5pyGroup
        self._id = f'{id(self._file)}/{self._zarr_array.name}'

        # See if we have the _COMPOUND_DTYPE attribute, which signifies that
        # this is a compound dtype
        compound_dtype_obj = _zarr_array.attrs.get("_COMPOUND_DTYPE", None)
        if compound_dtype_obj is not None:
            assert isinstance(compound_dtype_obj, list)
            # compound_dtype_obj is a list of tuples (name, dtype)
            # where dtype == "<REFERENCE>" if it represents an HDF5 reference
            for i in range(len(compound_dtype_obj)):
                if compound_dtype_obj[i][1] == '<REFERENCE>':
                    compound_dtype_obj[i][1] = h5py.special_dtype(ref=h5py.Reference)
            # If we have a compound dtype, then create the numpy dtype
            self._compound_dtype = np.dtype(
                [
                    (
                        compound_dtype_obj[i][0],
                        compound_dtype_obj[i][1]
                    )
                    for i in range(len(compound_dtype_obj))
                ]
            )
        else:
            self._compound_dtype = None

        # Check whether this is a scalar dataset
        self._is_scalar = self._zarr_array.attrs.get("_SCALAR", False)

        # The self._write object handles all the writing operations
        from .writers.LindiH5pyDatasetWriter import LindiH5pyDatasetWriter  # avoid circular import

        if self._readonly:
            self._writer = None
        else:
            self._writer = LindiH5pyDatasetWriter(self)

    @property
    def id(self):
        # see comment in LindiH5pyGroup
        return self._id

    @property
    def shape(self):  # type: ignore
        if self._is_scalar:
            return ()
        return self._zarr_array.shape

    @property
    def size(self):
        if self._is_scalar:
            return 1
        return self._zarr_array.size

    @property
    def dtype(self):
        if self._compound_dtype is not None:
            return self._compound_dtype
        ret = self._zarr_array.dtype
        if ret.kind == 'O':
            if not ret.metadata:
                # The following correction is needed because of
                # this code in hdmf/backends/hdf5/h5tools.py:
                # def _check_str_dtype(self, h5obj):
                #     dtype = h5obj.dtype
                #     if dtype.kind == 'O':
                #         if dtype.metadata.get('vlen') == str and H5PY_3:
                #             return StrDataset(h5obj, None)
                #     return h5obj
                # We cannot have a dtype with kind 'O' and no metadata
                # There is also this section in pynwb validator.py
                # if isinstance(received, np.dtype):
                #     if received.char == 'O':
                #         if 'vlen' in received.metadata:
                #             received = received.metadata['vlen']
                #         else:
                #             raise ValueError("Unrecognized type: '%s'" % received)
                #         received = 'utf' if received is str else 'ascii'
                #     elif received.char == 'U':
                #         received = 'utf'
                #     elif received.char == 'S':
                #         received = 'ascii'
                #     else:
                #         received = received.name
                # ------------------------------------------
                # I don't know how to figure out when vlen should be str or bytes
                # but validate seems to work only when I put in vlen = bytes
                #
                vlen = bytes
                ret = np.dtype(str(ret), metadata={'vlen': vlen})  # type: ignore
        return ret

    @property
    def nbytes(self):
        return self._zarr_array.nbytes

    @property
    def file(self):
        return self._file

    @property
    def name(self):
        return self._zarr_array.name

    @property
    def maxshape(self):
        # not sure what to return here, so let's return self.shape rather than self._h5py_dataset.maxshape
        # return self._h5py_dataset.maxshape
        return self.shape

    @property
    def ndim(self):
        if self._is_scalar:
            return 0
        return self._zarr_array.ndim

    @property
    def attrs(self):  # type: ignore
        return LindiH5pyAttributes(self._zarr_array.attrs, readonly=self._file.mode == 'r')

    @property
    def fletcher32(self):
        for f in self._zarr_array.filters:
            if f.__class__.__name__ == 'Fletcher32':
                return True
        return False

    @property
    def compression(self):
        # Return dummy value so we don't break hdmf
        return None

    @property
    def compression_opts(self):
        # Return dummy value so we don't break hdmf
        return None

    @property
    def chunks(self):
        return self._zarr_array.chunks

    def __repr__(self):  # type: ignore
        return f"<{self.__class__.__name__}: {self.name}>"

    def __str__(self):
        return f"<{self.__class__.__name__}: {self.name}>"

    def __getitem__(self, args, new_dtype=None):
        if new_dtype is not None:
            raise Exception("new_dtype is not supported for zarr.Array")
        options = self._file._range_read_options
        if options is None:
            return self._get_item_for_zarr(self._zarr_array, args)
        token, capture = start_range_read_query(options)
        try:
            result = self._get_item_for_zarr(self._zarr_array, args)
        except Exception:
            abort_range_read_query(token)
            raise
        complete_range_read_query(
            token,
            capture,
            dataset_path=self.name,
            zarr_array=self._zarr_array,
            result=result,
            warned_categories=self._file._range_efficiency_warned_categories,
            warning_lock=self._file._range_efficiency_warning_lock,
        )
        return result

    def __array__(self, dtype=None, copy=None):
        """Materialize through LINDI rather than h5py's low-level dataset ID."""
        array = np.asarray(self[()])
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        if copy is True:
            array = array.copy()
        return array

    def _get_item_for_zarr(self, zarr_array: zarr.Array, selection: Any):
        # First check whether this is an external array link
        external_array_link = zarr_array.attrs.get("_EXTERNAL_ARRAY_LINK", None)
        if external_array_link and isinstance(external_array_link, dict):
            link_type = external_array_link.get("link_type", None)
            if link_type == 'hdf5_dataset':
                url = external_array_link.get("url", None)
                name = external_array_link.get("name", None)
                if url is not None and name is not None:
                    client = self._get_external_hdf5_client(url)
                    dataset = client[name]
                    assert isinstance(dataset, h5py.Dataset)
                    return dataset[selection]
        if self._compound_dtype is not None:
            # Compound dtype
            # In this case we index into the compound dtype using the name of the field
            # For example, if the dtype is [('x', 'f4'), ('y', 'f4')], then we can do
            # dataset['x'][0] to get the first x value
            assert self._compound_dtype.names is not None
            if isinstance(selection, str):
                # Find the index of this field in the compound dtype
                ind = self._compound_dtype.names.index(selection)
                # Get the dtype of this field
                dt = self._compound_dtype[ind]
                if dt == 'object':
                    dtype = h5py.Reference
                else:
                    dtype = np.dtype(dt)
                # Return a new object that can be sliced further
                # It's important that the return type is Any here, because otherwise we get linter problems
                ret = LindiH5pyDatasetCompoundFieldSelection(
                    dataset=self, ind=ind, dtype=dtype
                )
                return ret
            compound_array = np.empty(zarr_array.shape, dtype=self._compound_dtype)
            encoded_array = zarr_array[:]
            for field_index, field_name in enumerate(self._compound_dtype.names):
                field_dtype = self._compound_dtype.fields[field_name][0]
                values = [value[field_index] for value in encoded_array.flat]
                if h5py.check_dtype(ref=field_dtype) is not None:
                    values = [
                        LindiH5pyReference(value['_REFERENCE'])
                        if isinstance(value, dict) and '_REFERENCE' in value
                        else value
                        for value in values
                    ]
                compound_array[field_name] = np.asarray(
                    values, dtype=field_dtype
                ).reshape(zarr_array.shape)
            return decode_references(compound_array[selection])

        # We use zarr's slicing, except in the case of a scalar dataset
        if self.ndim == 0:
            # make sure selection is ()
            if selection != ():
                raise TypeError(f'Cannot slice a scalar dataset with {selection}')
            # For some reason, with the newest version of zarr (2.18.0) we need to use [:][0] rather than just [0].
            # Otherwise we get an error "ValueError: buffer source array is read-only"
            return zarr_array[:][0]
        return decode_references(_read_zarr_selection(zarr_array, selection))

    def _get_external_hdf5_client(self, url: str) -> h5py.File:
        clients = self._file._external_hdf5_clients
        if url not in clients:
            if url.startswith("http://") or url.startswith("https://"):
                ff = LindiRemfile(url, local_cache=self._file._local_cache)
            elif url.startswith("file://"):
                parsed = urlsplit(url)
                if parsed.hostname not in (None, "", "localhost"):
                    raise ValueError(
                        f"Remote file URI authorities are not supported: {url}"
                    )
                if parsed.query or parsed.fragment:
                    raise ValueError(
                        f"File URI queries and fragments are not supported: {url}"
                    )
                ff = open(url2pathname(unquote(parsed.path)), "rb")
            else:
                ff = open(url, "rb")
            clients[url] = (h5py.File(ff, "r"), ff)
        return clients[url][0]

    @property
    def ref(self):
        if self._readonly:
            raise ValueError("Cannot get ref on read-only object")
        assert self._writer is not None
        return self._writer.ref

    ##############################
    # Write
    def __setitem__(self, args, val):
        if self._readonly:
            raise ValueError("Cannot set items on read-only object")
        assert self._writer is not None
        self._writer.__setitem__(args, val)


def _read_zarr_selection(zarr_array: zarr.Array, selection: Any):
    """Avoid a Zarr 2.18.2 integer-indexing bug on unit-width chunks."""
    original_selection = selection
    if not isinstance(selection, tuple):
        selection = (selection,)
    if any(x is None for x in selection):
        return zarr_array[original_selection]
    if sum(x is Ellipsis for x in selection) > 1:
        return zarr_array[original_selection]

    explicit_axis_count = sum(x is not Ellipsis for x in selection)
    missing_axis_count = zarr_array.ndim - explicit_axis_count
    if missing_axis_count < 0:
        return zarr_array[original_selection]
    expanded_selection = []
    for index in selection:
        if index is Ellipsis:
            expanded_selection.extend([slice(None)] * missing_axis_count)
        else:
            expanded_selection.append(index)
    if not any(x is Ellipsis for x in selection):
        expanded_selection.extend([slice(None)] * missing_axis_count)

    integer_axes = [
        axis
        for axis, index in enumerate(expanded_selection)
        if isinstance(index, (int, np.integer)) and not isinstance(index, (bool, np.bool_))
    ]
    if not integer_axes or not any(
        zarr_array.chunks[axis] == 1 for axis in integer_axes
    ):
        return zarr_array[original_selection]

    normalized_selection = list(expanded_selection)
    for axis in integer_axes:
        index = int(normalized_selection[axis])
        if index < 0:
            index += zarr_array.shape[axis]
        if index < 0 or index >= zarr_array.shape[axis]:
            return zarr_array[original_selection]
        normalized_selection[axis] = slice(index, index + 1)
    result = np.squeeze(
        zarr_array[tuple(normalized_selection)], axis=tuple(integer_axes)
    )
    # Match Zarr's native integer-indexing contract. In particular, callers such
    # as HDMF distinguish NumPy scalars from zero-dimensional ndarrays.
    return result[()] if result.ndim == 0 else result


class LindiH5pyDatasetCompoundFieldSelection:
    """
    This class is returned when a compound dataset is indexed with a field name.
    For example, if the dataset has dtype [('x', 'f4'), ('y', 'f4')], then we
    can do dataset['x'][0] to get the first x value. The dataset['x'] returns an
    object of this class.
    """
    def __init__(self, *, dataset: LindiH5pyDataset, ind: int, dtype: np.dtype):
        self._dataset = dataset  # The parent dataset
        self._ind = ind  # The index of the field in the compound dtype
        self._dtype = dtype  # The dtype of the field
        if self._dataset.ndim != 1:
            # For now we only support 1D datasets
            raise TypeError(
                f"Compound field selection only implemented for 1D datasets, not {self._dataset.ndim}D"
            )
        if not isinstance(self._dataset._zarr_array, zarr.Array):
            raise TypeError(
                f"Compound field selection only implemented for zarr.Array, not {type(self._dataset._zarr_array)}"
            )
        za = self._dataset._zarr_array
        self._zarr_array = za
        # Prepare the data in memory
        d = [za[i][self._ind] for i in range(len(za))]
        if self._dtype == h5py.Reference:
            # Convert to LindiH5pyReference
            # Every element in the selection should be a reference dict
            d = [LindiH5pyReference(x['_REFERENCE']) for x in d]
        self._data = np.array(d, dtype=self._dtype)

    def __len__(self):
        """We conform to h5py, which is the number of elements in the first dimension. TypeError if scalar"""
        if self.ndim == 0:
            raise TypeError("Scalar dataset")
        return self.shape[0]  # type: ignore

    def __iter__(self):
        """We conform to h5py, which is: Iterate over the first axis. TypeError if scalar."""
        shape = self.shape
        if len(shape) == 0:
            raise TypeError("Can't iterate over a scalar dataset")
        for i in range(shape[0]):
            yield self[i]

    @property
    def ndim(self):
        return self._zarr_array.ndim

    @property
    def shape(self):
        return self._zarr_array.shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def size(self):
        return self._data.size

    def __getitem__(self, selection):
        return decode_references(self._data[selection])
