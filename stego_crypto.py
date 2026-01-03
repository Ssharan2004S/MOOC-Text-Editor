# stego_crypto.py
"""
Complete stego + crypto helpers used by cryptAVIT Flask app.

Provides:
- RSA keypair generation (PEM)
- Hybrid encrypt/decrypt (AES-GCM + RSA-OAEP)
- Packaging (binary container with MAGIC)
- LSB embed/extract for images (PNG/BMP), WAV audio (16-bit PCM)
- Zero-width text stego
- Simple video fallback (append package)
"""

import io
import os
import json
import struct
import zlib
import secrets
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import serialization, hashes
from PIL import Image
import numpy as np
import wave

MAGIC = b'CRYPTAV1'  # 8 bytes marker

# ---------------- RSA key generation ----------------
def generate_rsa_keypair_pem(password: bytes = None):
    """
    Generate RSA private/public key pair and return (priv_pem, pub_pem) as bytes.
    If password provided (bytes), private key is encrypted with that passphrase.
    """
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    public_key = private_key.public_key()
    enc = serialization.BestAvailableEncryption(password) if password else serialization.NoEncryption()
    priv_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=enc,
    )
    pub_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return priv_pem, pub_pem

# ---------------- Pack/Unpack package ----------------
def build_package(wrapped_key: bytes, nonce: bytes, ciphertext: bytes, meta: dict = None):
    meta = meta or {}
    meta_json = json.dumps(meta, separators=(',', ':')).encode('utf-8')
    buf = io.BytesIO()
    buf.write(MAGIC)
    buf.write(struct.pack('>H', len(meta_json)))
    buf.write(meta_json)
    buf.write(struct.pack('>H', len(wrapped_key)))
    buf.write(wrapped_key)
    buf.write(nonce)
    buf.write(struct.pack('>I', len(ciphertext)))
    buf.write(ciphertext)
    return buf.getvalue()

def parse_package(data: bytes):
    bio = io.BytesIO(data)
    magic = bio.read(len(MAGIC))
    if magic != MAGIC:
        raise ValueError('Not a cryptAVIT package (magic mismatch)')
    meta_len = struct.unpack('>H', bio.read(2))[0]
    meta_json = bio.read(meta_len)
    meta = json.loads(meta_json.decode('utf-8')) if meta_json else {}
    wk_len = struct.unpack('>H', bio.read(2))[0]
    wrapped_key = bio.read(wk_len)
    nonce = bio.read(12)
    ct_len = struct.unpack('>I', bio.read(4))[0]
    ciphertext = bio.read(ct_len)
    return wrapped_key, nonce, ciphertext, meta

# ---------------- Hybrid encrypt/decrypt ----------------
def hybrid_encrypt_package(plaintext: bytes, recipient_pub_pem: bytes, compress: bool = False):
    """
    Encrypt plaintext with AES-GCM and wrap AES key with recipient RSA public key (OAEP).
    Returns packaged bytes ready for embedding.
    """
    if compress:
        plaintext = zlib.compress(plaintext)
    aes_key = AESGCM.generate_key(bit_length=256)
    aesgcm = AESGCM(aes_key)
    nonce = os.urandom(12)
    ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data=None)
    pub = serialization.load_pem_public_key(recipient_pub_pem)
    wrapped = pub.encrypt(
        aes_key,
        padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None),
    )
    meta = {'alg': 'AES-GCM-256+RSA-OAEP', 'compressed': bool(compress)}
    return build_package(wrapped, nonce, ciphertext, meta)

def hybrid_decrypt_package(package_bytes: bytes, recipient_priv_pem: bytes, password: bytes = None):
    """
    Parse package, unwrap AES key with recipient private key, decrypt AES-GCM ciphertext.
    Return plaintext bytes (decompressed if meta says so).
    """
    wrapped_key, nonce, ciphertext, meta = parse_package(package_bytes)
    priv = serialization.load_pem_private_key(recipient_priv_pem, password=None)
    aes_key = priv.decrypt(
        wrapped_key,
        padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None),
    )
    aesgcm = AESGCM(aes_key)
    plaintext = aesgcm.decrypt(nonce, ciphertext, associated_data=None)
    if meta.get('compressed'):
        plaintext = zlib.decompress(plaintext)
    return plaintext

# ---------------- Bit utilities ----------------
def _bytes_to_bitarray(b: bytes, total_bits: int):
    src = np.frombuffer(b, dtype=np.uint8)
    bits = np.unpackbits(src)  # MSB-first per byte
    if bits.size < total_bits:
        pad = np.zeros(total_bits - bits.size, dtype=np.uint8)
        bits = np.concatenate((bits, pad))
    elif bits.size > total_bits:
        bits = bits[:total_bits]
    return bits.astype(np.uint8)

def _bitarray_to_bytes(bits: np.ndarray):
    if bits.size % 8 != 0:
        pad_len = 8 - (bits.size % 8)
        bits = np.concatenate((bits, np.zeros(pad_len, dtype=np.uint8)))
    packed = np.packbits(bits)
    return packed.tobytes()

# ---------------- LSB helpers for image (PNG/BMP) - optimized ----------------
def embed_into_image_lsb(image_file_like, package_bytes: bytes, lsb_bits: int = 1):
    """
    Embed package bytes into RGBA image LSBs. Returns PNG bytes.
    """
    if lsb_bits < 1 or lsb_bits > 4:
        raise ValueError("lsb_bits must be 1..4")
    image = Image.open(image_file_like).convert('RGBA')
    arr = np.array(image, dtype=np.uint8)  # shape (h,w,4)
    flat = arr.ravel()
    total_capacity_bits = flat.size * lsb_bits
    need_bits = len(package_bytes) * 8
    if need_bits > total_capacity_bits:
        raise ValueError(f'Package too large for this image (need {need_bits} bits, capacity {total_capacity_bits} bits)')
    bits = _bytes_to_bitarray(package_bytes, total_capacity_bits)  # length total_capacity_bits
    bits_per_value = bits.reshape(-1, lsb_bits)[:flat.size, :]
    shifts = (1 << np.arange(lsb_bits, dtype=np.uint8))
    delta = (bits_per_value * shifts).sum(axis=1).astype(np.uint8)
    mask = (~((1 << lsb_bits) - 1)) & 0xFF
    flat = (flat & mask) | delta
    out_arr = flat.reshape(arr.shape)
    out_img = Image.fromarray(out_arr, mode='RGBA')
    buf = io.BytesIO()
    out_img.save(buf, format='PNG', optimize=False, compress_level=1)
    return buf.getvalue()

def extract_from_image_lsb(image_file_like, lsb_bits: int = 1):
    """
    Extract package bytes from an RGBA image LSBs and return the package bytes starting at MAGIC.
    """
    if lsb_bits < 1 or lsb_bits > 4:
        raise ValueError("lsb_bits must be 1..4")
    image = Image.open(image_file_like).convert('RGBA')
    arr = np.array(image, dtype=np.uint8)
    flat = arr.ravel()
    lower = flat & ((1 << lsb_bits) - 1)
    lower_bits = ((lower[:, None] >> np.arange(lsb_bits, dtype=np.uint8)) & 1).astype(np.uint8)
    bits = lower_bits.reshape(-1)
    data = _bitarray_to_bytes(bits)
    i = data.find(MAGIC)
    if i == -1:
        raise ValueError('No cryptAVIT package found in image')
    return data[i:]

# ---------------- Robust WAV LSB embed/extract (16-bit PCM) ----------------
def embed_into_wav_lsb(wav_file_like, package_bytes: bytes, lsb_bits: int = 1):
    """
    Embed package_bytes into a 16-bit PCM WAV (any channels) using lsb_bits LSBs per sample.
    Returns bytes of the modified WAV file.
    """
    if lsb_bits < 1 or lsb_bits > 4:
        raise ValueError("lsb_bits must be 1..4")
    wav_in = wave.open(wav_file_like, 'rb')
    params = wav_in.getparams()  # (nchannels, sampwidth, framerate, nframes, comptype, compname)
    nchannels, sampwidth, framerate, nframes, comptype, compname = params
    frames = wav_in.readframes(nframes)
    wav_in.close()
    if sampwidth != 2:
        raise ValueError(f"Only 16-bit PCM WAV supported (sampwidth=2). Uploaded file has sampwidth={sampwidth}")
    samples = np.frombuffer(frames, dtype='<i2').astype(np.int16).copy()
    total_samples = samples.size
    total_capacity_bits = total_samples * lsb_bits
    need_bits = len(package_bytes) * 8
    if need_bits > total_capacity_bits:
        raise ValueError(f'Package too large for this WAV (need {need_bits} bits, capacity {total_capacity_bits} bits)')
    # Prepare bits
    src = np.frombuffer(package_bytes, dtype=np.uint8)
    bits = np.unpackbits(src)
    if bits.size < total_capacity_bits:
        pad = np.zeros(total_capacity_bits - bits.size, dtype=np.uint8)
        bits = np.concatenate((bits, pad))
    else:
        bits = bits[:total_capacity_bits]
    bits_per_sample = bits.reshape(-1, lsb_bits)[:total_samples, :]
    samples_u16 = samples.view(np.uint16)
    shifts = (1 << np.arange(lsb_bits, dtype=np.uint16))
    delta = (bits_per_sample * shifts).sum(axis=1).astype(np.uint16)
    mask = (~((1 << lsb_bits) - 1)) & 0xFFFF
    samples_u16 = (samples_u16 & mask) | delta
    out_frames = samples_u16.astype('<u2').tobytes()
    out_buf = io.BytesIO()
    wav_out = wave.open(out_buf, 'wb')
    wav_out.setparams(params)
    wav_out.writeframes(out_frames)
    wav_out.close()
    return out_buf.getvalue()

def extract_from_wav_lsb(wav_file_like, lsb_bits: int = 1):
    """
    Extract a cryptAVIT package from a 16-bit PCM WAV (any channels) and return the package bytes.
    """
    if lsb_bits < 1 or lsb_bits > 4:
        raise ValueError("lsb_bits must be 1..4")
    wav_in = wave.open(wav_file_like, 'rb')
    params = wav_in.getparams()
    nchannels, sampwidth, framerate, nframes, comptype, compname = params
    frames = wav_in.readframes(nframes)
    wav_in.close()
    if sampwidth != 2:
        raise ValueError(f"Only 16-bit PCM WAV supported (sampwidth=2). Uploaded file has sampwidth={sampwidth}")
    samples = np.frombuffer(frames, dtype='<i2')
    total_samples = samples.size
    samples_u16 = samples.view(np.uint16)
    lower = samples_u16 & ((1 << lsb_bits) - 1)
    lower_bits = ((lower[:, None] >> np.arange(lsb_bits, dtype=np.uint8)) & 1).astype(np.uint8)
    bits = lower_bits.reshape(-1)
    if bits.size % 8 != 0:
        pad = np.zeros(8 - (bits.size % 8), dtype=np.uint8)
        bits = np.concatenate((bits, pad))
    packed = np.packbits(bits)
    data = packed.tobytes()
    idx = data.find(MAGIC)
    if idx == -1:
        raise ValueError('No cryptAVIT package found in WAV')
    return data[idx:]

# ---------------- Zero-width text embed/extract ----------------
ZERO = '\u200b'  # zero-width space -> bit 0
ONE = '\u200c'   # zero-width non-joiner -> bit 1

def embed_into_text_zero_width(text: str, package_bytes: bytes):
    bits = _bytes_to_bitarray(package_bytes, len(package_bytes)*8)
    lines = text.splitlines(True)
    out = []
    idx = 0
    for line in lines:
        out.append(line.rstrip('\n'))
        if idx < bits.size:
            out.append(ZERO if bits[idx] == 0 else ONE)
            idx += 1
        if line.endswith('\n'):
            out.append('\n')
    while idx < bits.size:
        out.append(ZERO if bits[idx] == 0 else ONE)
        idx += 1
    return ''.join(out)

def extract_from_text_zero_width(text: str):
    chars = [c for c in text if c in (ZERO, ONE)]
    if not chars:
        raise ValueError('No hidden zero-width characters found')
    bits = np.array([0 if c == ZERO else 1 for c in chars], dtype=np.uint8)
    data = _bitarray_to_bytes(bits)
    i = data.find(MAGIC)
    if i == -1:
        raise ValueError('No cryptAVIT package found in text')
    return data[i:]

# ---------------- Video fallback (append package to file end) ----------------
def embed_into_video_fallback(video_file_like, package_bytes: bytes):
    data = video_file_like.read()
    return data + package_bytes

def extract_from_video_fallback(video_file_like):
    data = video_file_like.read()
    idx = data.find(MAGIC)
    if idx == -1:
        raise ValueError('No cryptAVIT package found in video')
    return data[idx:]
