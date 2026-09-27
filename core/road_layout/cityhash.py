"""CityHash64 from CityHash 1.0.3.

SCS archives hash paths with this revision. The 1.1 CityHash64, which is what
current PyPI builds ship, does not match those hashes. The empty string is the
one value both revisions share, so a newer hash looks fine until the first
real path misses.

Ported from CityHash 1.0.3 (Google, MIT). See THIRD_PARTY_LICENSES/cityhash-LICENSE.txt.
"""

from __future__ import annotations

_MASK = (1 << 64) - 1

_K0 = 0xC3A5C85C97CB3127
_K1 = 0xB492B66FBE98F273
_K2 = 0x9AE16A3B2F90404F
_K3 = 0xC949D7C7509E6557
_KMUL = 0x9DDFEA08EB382D69


def _u(n: int) -> int:
    return n & _MASK


def _fetch64(buf: bytes, i: int) -> int:
    return int.from_bytes(buf[i:i + 8], "little")


def _fetch32(buf: bytes, i: int) -> int:
    return int.from_bytes(buf[i:i + 4], "little")


def _rot(val: int, shift: int) -> int:
    val = _u(val)
    if shift == 0:
        return val
    shift &= 63
    return _u((val >> shift) | (val << (64 - shift)))


def _shift_mix(val: int) -> int:
    val = _u(val)
    return val ^ (val >> 47)


def _hash128to64(lo: int, hi: int) -> int:
    a = _u((lo ^ hi) * _KMUL)
    a ^= a >> 47
    b = _u((hi ^ a) * _KMUL)
    b ^= b >> 47
    return _u(b * _KMUL)


def _hash_len16(u: int, v: int) -> int:
    return _hash128to64(_u(u), _u(v))


def _hash_len_0_to_16(buf: bytes, length: int) -> int:
    if length > 8:
        a = _fetch64(buf, 0)
        b = _fetch64(buf, length - 8)
        return _hash_len16(a, _rot(b + length, length)) ^ b
    if length >= 4:
        a = _fetch32(buf, 0)
        return _hash_len16(length + (a << 3), _fetch32(buf, length - 4))
    if length > 0:
        a = buf[0]
        b = buf[length >> 1]
        c = buf[length - 1]
        y = a + (b << 8)
        z = length + (c << 2)
        return _u(_shift_mix(_u(y * _K2) ^ _u(z * _K3)) * _K2)
    return _K2


def _hash_len_17_to_32(buf: bytes, length: int) -> int:
    a = _u(_fetch64(buf, 0) * _K1)
    b = _fetch64(buf, 8)
    c = _u(_fetch64(buf, length - 8) * _K2)
    d = _u(_fetch64(buf, length - 16) * _K0)
    return _hash_len16(
        _u(_rot(a - b, 43) + _rot(c, 30) + d),
        _u(a + _rot(b ^ _K3, 20) - c + length),
    )


def _weak32(w: int, x: int, y: int, z: int, a: int, b: int) -> tuple[int, int]:
    a = _u(a + w)
    b = _rot(b + a + z, 21)
    c = a
    a = _u(a + x)
    a = _u(a + y)
    b = _u(b + _rot(a, 44))
    return _u(a + z), _u(b + c)


def _weak32_at(buf: bytes, i: int, a: int, b: int) -> tuple[int, int]:
    return _weak32(
        _fetch64(buf, i),
        _fetch64(buf, i + 8),
        _fetch64(buf, i + 16),
        _fetch64(buf, i + 24),
        a,
        b,
    )


def _hash_len_33_to_64(buf: bytes, length: int) -> int:
    z = _fetch64(buf, 24)
    a = _u(_fetch64(buf, 0) + _u((length + _fetch64(buf, length - 16)) * _K0))
    b = _rot(a + z, 52)
    c = _rot(a, 37)
    a = _u(a + _fetch64(buf, 8))
    c = _u(c + _rot(a, 7))
    a = _u(a + _fetch64(buf, 16))
    vf = _u(a + z)
    vs = _u(b + _rot(a, 31) + c)
    a = _u(_fetch64(buf, 16) + _fetch64(buf, length - 32))
    z = _fetch64(buf, length - 8)
    b = _rot(a + z, 52)
    c = _rot(a, 37)
    a = _u(a + _fetch64(buf, length - 24))
    c = _u(c + _rot(a, 7))
    a = _u(a + _fetch64(buf, length - 16))
    wf = _u(a + z)
    ws = _u(b + _rot(a, 31) + c)
    r = _shift_mix(_u((vf + ws) * _K2) + _u((wf + vs) * _K0))
    return _u(_shift_mix(_u(r * _K0) + vs) * _K2)


def city_hash64(data: bytes) -> int:
    """CityHash64 of *data*, as the game hashes an archive path."""
    length = len(data)
    if length <= 32:
        if length <= 16:
            return _u(_hash_len_0_to_16(data, length))
        return _u(_hash_len_17_to_32(data, length))
    if length <= 64:
        return _u(_hash_len_33_to_64(data, length))

    x = _fetch64(data, length - 40)
    y = _u(_fetch64(data, length - 16) + _fetch64(data, length - 56))
    z = _hash_len16(_u(_fetch64(data, length - 48) + length), _fetch64(data, length - 24))
    v0, v1 = _weak32_at(data, length - 64, length, z)
    w0, w1 = _weak32_at(data, length - 32, _u(y + _K1), x)
    x = _u(x * _K1 + _fetch64(data, 0))

    # Walk 64-byte chunks from the front. `end` is the byte the loop stops on.
    end = (length - 1) & ~63
    off = 0
    while end != 0:
        x = _u(_rot(x + y + v0 + _fetch64(data, off + 8), 37) * _K1)
        y = _u(_rot(y + v1 + _fetch64(data, off + 48), 42) * _K1)
        x ^= w1
        y = _u(y + v0 + _fetch64(data, off + 40))
        z = _u(_rot(z + w0, 33) * _K1)
        v0, v1 = _weak32_at(data, off, _u(v1 * _K1), _u(x + w0))
        w0, w1 = _weak32_at(data, off + 32, _u(z + w1), _u(y + _fetch64(data, off + 16)))
        z, x = x, z
        off += 64
        end -= 64
    return _hash_len16(
        _u(_hash_len16(v0, w0) + _u(_shift_mix(y) * _K1) + z),
        _u(_hash_len16(v1, w1) + x),
    )


def city_hash64_path(path: str) -> int:
    """Hash an archive key. Keys are ASCII, no leading slash."""
    return city_hash64(path.encode("utf-8"))
