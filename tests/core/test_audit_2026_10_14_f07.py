"""The 2026-10-14 audit, fixer F07: the shared ``core/safeio`` leaves.

Findings packwright-16 (atomic's rename), shell-documents-01 (zero-width
dtypes), shell-documents-07 (corrupt npz and rblk members), inker-mode-12
(Pillow's own bomb error) and shell-documents-02 (the bounded read's up-front
allocation).
"""

from __future__ import annotations

import io
import os
import struct
import tracemalloc
import zipfile

import numpy as np
import pytest
from PIL import Image

from realmspinner.core.safeio import atomic, npyguard, pixelguard, sizeguard
from realmspinner.service.errors import TooLarge


def _flaky_replace(monkeypatch, failures: int) -> list[int]:
    """``os.replace`` that raises WinError 5 *failures* times, then works."""
    real = os.replace
    calls: list[int] = []

    def replace(src, dst):
        calls.append(1)
        if len(calls) <= failures:
            raise PermissionError(13, "Access is denied", str(dst))
        return real(src, dst)

    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(atomic, "REPLACE_PAUSE", 0.0, raising=False)
    return calls


def test_staged_rides_out_a_destination_handle_that_closes_within_a_moment(
    tmp_path, monkeypatch
) -> None:
    dest = tmp_path / "export.bin"
    dest.write_bytes(b"old")
    calls = _flaky_replace(monkeypatch, 2)
    atomic.write_bytes(dest, b"new")
    assert dest.read_bytes() == b"new"
    assert len(calls) == 3
    assert list(tmp_path.iterdir()) == [dest]


def test_staged_set_rides_out_a_destination_handle_that_closes_within_a_moment(
    tmp_path, monkeypatch
) -> None:
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"old")
    _flaky_replace(monkeypatch, 2)
    atomic.staged_set({a: b"A", b: b"B"})
    assert (a.read_bytes(), b.read_bytes()) == (b"A", b"B")


def test_staged_still_raises_when_the_handle_never_closes(tmp_path, monkeypatch) -> None:
    dest = tmp_path / "export.bin"
    dest.write_bytes(b"old")
    _flaky_replace(monkeypatch, 10**6)
    with pytest.raises(PermissionError):
        atomic.write_bytes(dest, b"new")
    assert dest.read_bytes() == b"old"
    assert [p.name for p in tmp_path.iterdir()] == ["export.bin"]


def _header(shape, descr: str) -> bytes:
    d = f"{{'descr': {descr!r}, 'fortran_order': False, 'shape': {shape!r}, }}".encode("latin1")
    pad = (64 - (10 + len(d) + 1) % 64) % 64
    d = d + b" " * pad + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(d)) + d


@pytest.mark.parametrize("descr", ["|V0", "<U0", "|S0"])
def test_an_npy_declaring_a_zero_width_dtype_with_a_huge_shape_is_refused_by_name(
    descr,
) -> None:
    with pytest.raises(ValueError, match="zero-width"):
        npyguard.read_array(_header((1 << 40,), descr), "a layer")


def _npy(a: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.lib.format.write_array(buf, a, allow_pickle=False)
    return buf.getvalue()


def _zip(members: dict[str, bytes], method=zipfile.ZIP_DEFLATED) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", method) as z:
        for name, blob in members.items():
            z.writestr(name, blob)
    return buf.getvalue()


def test_a_corrupt_npz_member_is_refused_as_a_value_error() -> None:
    good = _npy(np.arange(64, dtype=np.int32))
    stored = bytearray(_zip({"a.npy": good}, zipfile.ZIP_STORED))
    stored[30 + 5 + len(good) // 2] ^= 0xFF  # a CRC failure
    with pytest.raises(ValueError):
        npyguard.read_npz(bytes(stored), "z")

    deflated = bytearray(_zip({"a.npy": good}))
    deflated[30 + 5 + 4] ^= 0xFF  # a damaged deflate stream
    with pytest.raises(ValueError):
        npyguard.read_npz(bytes(deflated), "z")

    encrypted = bytearray(_zip({"a.npy": good}))
    encrypted[6] |= 1
    encrypted[encrypted.rfind(b"PK\x01\x02") + 8] |= 1
    with pytest.raises(ValueError):
        npyguard.read_npz(bytes(encrypted), "z")

    unknown = bytearray(_zip({"a.npy": good}))
    unknown[8] = 99
    unknown[unknown.rfind(b"PK\x01\x02") + 10] = 99
    with pytest.raises(ValueError):
        npyguard.read_npz(bytes(unknown), "z")


def test_a_corrupt_rblk_mesh_member_is_refused_as_a_value_error() -> None:
    from realmspinner.kernels.mesh import document as bd
    from realmspinner.kernels.mesh import primitives as bp
    from realmspinner.kernels.mesh import serialize

    doc = bd.ClayDoc()
    doc.objects.append(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box(), material=0))
    data = serialize.rblk_bytes(doc)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        member = next(i for i in zf.infolist() if i.filename.endswith(".npz"))
        start = member.header_offset + 30 + len(member.filename) + len(member.extra)
    # A flipped byte inside the member's data: a CRC mismatch when stored, a
    # damaged stream (zlib.error) when deflated -- either leaked unnamed.
    blob = bytearray(data)
    blob[start + member.compress_size // 2] ^= 0xFF
    with pytest.raises(ValueError):
        serialize.read_rblk(bytes(blob))
    # An unknown compression method on the same member (NotImplementedError).
    blob = bytearray(data)
    blob[member.header_offset + 8] = 99
    cd = blob.find(b"PK" + b"", 0)
    while cd != -1:
        name_len = struct.unpack_from("<H", blob, cd + 28)[0]
        if bytes(blob[cd + 46 : cd + 46 + name_len]).decode() == member.filename:
            blob[cd + 10] = 99
            break
        cd = blob.find(b"PK", cd + 4)
    with pytest.raises(ValueError):
        serialize.read_rblk(bytes(blob))


def test_an_image_past_twice_the_ceiling_is_refused_by_name(monkeypatch) -> None:
    monkeypatch.setattr(pixelguard, "MAX_DECODE_PIXELS", 100)
    monkeypatch.setattr(pixelguard, "_armed", True)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    buf = io.BytesIO()
    Image.new("RGB", (20, 20)).save(buf, "PNG")  # 400 px > 2 x 100: Pillow raises
    with pytest.raises(ValueError), pixelguard.opened(io.BytesIO(buf.getvalue()), "a picture"):
        pytest.fail("a refused image must not be yielded")
    with pytest.raises(ValueError):
        pixelguard.decode_rgba(io.BytesIO(buf.getvalue()), "a picture")


def test_read_bytes_within_ceiling_allocates_in_proportion_to_the_file_not_the_ceiling(
    tmp_path,
) -> None:
    path = tmp_path / "tiny.bin"
    path.write_bytes(b"x" * 1024)
    ceiling = 1 << 28
    tracemalloc.start()
    try:
        data = sizeguard.read_bytes_within_ceiling(path, ceiling)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert data == b"x" * 1024
    assert peak < 1 << 20, f"peaked at {peak} bytes opening a 1 KB file"


def test_read_bytes_within_ceiling_still_refuses_a_file_that_grew_after_the_stat(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "grows.bin"
    path.write_bytes(b"x" * 20)
    real = os.fstat

    class _Stale:
        st_size = 5  # what the stat saw before the writer appended

    monkeypatch.setattr(os, "fstat", lambda fd: _Stale() if fd >= 0 else real(fd))
    with pytest.raises(TooLarge):
        sizeguard.read_bytes_within_ceiling(path, 10)
    path.write_bytes(b"x" * 8)
    assert sizeguard.read_bytes_within_ceiling(path, 10) == b"x" * 8
