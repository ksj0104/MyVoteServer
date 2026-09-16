"""Explicit local research PKI provisioning and bounded mTLS material loading.

No network access or system trust-store registration. Private PEM keys are stored
unencrypted inside a current-user-only directory for unattended local workers.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import ipaddress
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Literal


MAX_PEM_BYTES = 64 * 1024


@dataclass(frozen=True)
class PkiPaths:
    directory: Path
    ca_cert: Path
    ca_key: Path
    server_cert: Path
    server_key: Path
    client_cert: Path
    client_key: Path
    manifest: Path


@dataclass(frozen=True)
class TlsMaterial:
    certificate_chain: bytes = field(repr=False)
    private_key: bytes = field(repr=False)
    root_certificates: bytes = field(repr=False)


def _crypto():
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
    except ImportError as exc:
        raise RuntimeError("install optional cryptography for gateway certificate provisioning") from exc
    return x509, hashes, serialization, ec, ExtendedKeyUsageOID, NameOID


def _windows_sid() -> str:
    result = subprocess.run(["whoami.exe", "/user", "/fo", "csv", "/nh"],
                            capture_output=True, check=True, timeout=10,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    match = re.search(rb"S-1-[0-9-]+", result.stdout)
    if match is None:
        raise PermissionError("cannot determine current Windows user SID")
    return match.group().decode("ascii")


def _windows_dacl(path: Path) -> str:
    """Read a DACL as SDDL through documented Windows security APIs."""
    import ctypes
    from ctypes import wintypes
    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    get = advapi.GetNamedSecurityInfoW
    get.argtypes = [wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    get.restype = wintypes.DWORD
    convert = advapi.ConvertSecurityDescriptorToStringSecurityDescriptorW
    convert.argtypes = [ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
                        ctypes.POINTER(wintypes.LPWSTR), ctypes.c_void_p]
    convert.restype = wintypes.BOOL
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    descriptor = ctypes.c_void_p()
    result = get(str(path), 1, 4, None, None, None, None, ctypes.byref(descriptor))
    if result:
        raise PermissionError(f"cannot inspect Windows TLS file DACL (code {result})")
    text = wintypes.LPWSTR()
    try:
        if not convert(descriptor, 1, 4, ctypes.byref(text), None):
            raise PermissionError("cannot convert Windows TLS file DACL")
        return text.value
    finally:
        if text:
            kernel.LocalFree(ctypes.cast(text, ctypes.c_void_p))
        kernel.LocalFree(descriptor)


def _check_private_permissions(path: Path) -> None:
    if os.name == "nt":
        sid = _windows_sid()
        dacl = _windows_dacl(path)
        # Provisioned files inherit exactly one current-user full-control ACE.
        # Reject copied material exposing the key to any additional principal.
        entries = re.findall(r"\(([^()]*)\)", dacl)
        if not entries or any(len(item.split(";")) != 6
                              or item.split(";")[0] != "A"
                              or item.split(";")[-1] != sid for item in entries):
            raise PermissionError("TLS private key must have a current-user-only Windows DACL")
    else:
        mode = path.stat()
        if mode.st_uid != os.getuid() or stat.S_IMODE(mode.st_mode) & 0o077:
            raise PermissionError("TLS private key must belong to the current user with mode 0600")


def _restrict_windows_directory(path: Path) -> None:
    import ctypes
    from ctypes import wintypes
    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    convert = advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW
    convert.argtypes = [wintypes.LPCWSTR, wintypes.DWORD,
                        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    convert.restype = wintypes.BOOL
    get_dacl = advapi.GetSecurityDescriptorDacl
    get_dacl.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.BOOL),
                        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(wintypes.BOOL)]
    get_dacl.restype = wintypes.BOOL
    set_acl = advapi.SetNamedSecurityInfoW
    set_acl.argtypes = [wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD,
                       ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    set_acl.restype = wintypes.DWORD
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    descriptor = ctypes.c_void_p()
    # Replace the full DACL, including explicit entries created by Windows/Python,
    # rather than merely removing inherited permissions with icacls.
    sddl = f"D:P(A;OICI;FA;;;{_windows_sid()})"
    if not convert(sddl, 1, ctypes.byref(descriptor), None):
        raise PermissionError("cannot construct current-user-only Windows DACL")
    try:
        present, defaulted = wintypes.BOOL(), wintypes.BOOL()
        dacl = ctypes.c_void_p()
        if not get_dacl(descriptor, ctypes.byref(present), ctypes.byref(dacl), ctypes.byref(defaulted)) or not present:
            raise PermissionError("cannot obtain current-user-only Windows DACL")
        status = set_acl(str(path), 1, 4 | 0x80000000, None, None, dacl, None)
        if status:
            raise PermissionError(f"cannot protect Windows TLS directory (code {status})")
    finally:
        kernel.LocalFree(descriptor)


def _private_directory(path: Path) -> None:
    # A fresh directory is necessary: do not repurpose or tighten permissions on
    # the operator's existing directory. No keys exist until protection succeeds.
    path.mkdir(mode=0o700, parents=False, exist_ok=False)
    if os.name == "nt":
        _restrict_windows_directory(path)
        _check_private_permissions(path)
    else:
        os.chmod(path, 0o700)


def _write_new(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as destination:
        destination.write(data)
    if os.name != "nt":
        os.chmod(path, 0o600)


def _server_names(names) -> tuple[str, ...]:
    if isinstance(names, str):
        raise ValueError("server_names must be a sequence of explicit DNS names or IP addresses")
    values = tuple(names)
    if not 1 <= len(values) <= 16:
        raise ValueError("provide one to sixteen explicit server names")
    result = []
    for value in values:
        if not isinstance(value, str) or not value or value.strip() != value or "%" in value:
            raise ValueError("invalid server name")
        try:
            canonical = str(ipaddress.ip_address(value))
        except ValueError:
            try:
                canonical = value.rstrip(".").encode("idna").decode("ascii").lower()
            except UnicodeError as exc:
                raise ValueError("invalid DNS name") from exc
            labels = canonical.split(".")
            if (len(canonical) > 253 or not all(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
                                               for label in labels)):
                raise ValueError("use a DNS name or bare IP address, without wildcard, URL, port, or path")
        if canonical not in result:
            result.append(canonical)
    return tuple(result)


def create_local_pki(output_dir: str | Path, *, server_names: tuple[str, ...],
                     client_id: str, valid_days: int = 30) -> PkiPaths:
    """Generate a new CA and one server/client identity in a NEW local directory."""
    names = _server_names(server_names)
    if not isinstance(client_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", client_id):
        raise ValueError("client_id must contain 1-64 ASCII letters, digits, dots, underscores or hyphens")
    if isinstance(valid_days, bool) or not isinstance(valid_days, int) or not 1 <= valid_days <= 365:
        raise ValueError("valid_days must be an integer in [1, 365]")
    directory = Path(output_dir).expanduser().absolute()
    if directory.exists() or directory.is_symlink():
        raise FileExistsError("TLS output directory already exists; choose a new directory")
    if not directory.parent.is_dir():
        raise FileNotFoundError("create the TLS output parent directory first")
    x509, hashes, serialization, ec, eku, oid = _crypto()
    start = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=5)
    end = start + timedelta(days=valid_days)
    keys = {role: ec.generate_private_key(ec.SECP256R1()) for role in ("ca", "server", "client")}
    ca_name = x509.Name([x509.NameAttribute(oid.COMMON_NAME, "MyVote Local Research CA")])

    def base(subject, issuer, public_key):
        return (x509.CertificateBuilder().subject_name(subject).issuer_name(issuer)
                .public_key(public_key).serial_number(x509.random_serial_number())
                .not_valid_before(start).not_valid_after(end))

    def key_usage(ca):
        return x509.KeyUsage(digital_signature=not ca, content_commitment=False,
                             key_encipherment=False, data_encipherment=False,
                             key_agreement=False, key_cert_sign=ca, crl_sign=ca,
                             encipher_only=False, decipher_only=False)

    ca = (base(ca_name, ca_name, keys["ca"].public_key())
          .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
          .add_extension(key_usage(True), critical=True)
          .add_extension(x509.SubjectKeyIdentifier.from_public_key(keys["ca"].public_key()), critical=False)
          .sign(keys["ca"], hashes.SHA256()))
    certificates = {"ca": ca}
    for role in ("server", "client"):
        # A generic server CN avoids encoding long DNS names into a 64-byte CN.
        subject = x509.Name([x509.NameAttribute(oid.COMMON_NAME,
                            "MyVote Gateway" if role == "server" else client_id)])
        if role == "server":
            sans = []
            for name in names:
                try:
                    sans.append(x509.IPAddress(ipaddress.ip_address(name)))
                except ValueError:
                    sans.append(x509.DNSName(name))
        else:
            sans = [x509.UniformResourceIdentifier("urn:myvote:client:" + client_id)]
        certificates[role] = (
            base(subject, ca_name, keys[role].public_key())
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(key_usage(False), critical=True)
            .add_extension(x509.ExtendedKeyUsage([eku.SERVER_AUTH if role == "server" else eku.CLIENT_AUTH]), critical=False)
            .add_extension(x509.SubjectAlternativeName(sans), critical=False)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(keys[role].public_key()), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(keys["ca"].public_key()), critical=False)
            .sign(keys["ca"], hashes.SHA256()))
    _private_directory(directory)
    for role in ("ca", "server", "client"):
        _write_new(directory / f"{role}.key", keys[role].private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption()))
        _check_private_permissions(directory / f"{role}.key")
        _write_new(directory / f"{role}.crt", certificates[role].public_bytes(serialization.Encoding.PEM))
    manifest = {
        "schema_version": 1, "purpose": "myvote-local-research-mtls",
        "server_names": list(names), "client_id": client_id,
        "not_valid_before_utc": start.isoformat(), "not_valid_after_utc": end.isoformat(),
        "key_algorithm": "ECDSA P-256", "signature_hash": "SHA256",
        "certificate_sha256": {f"{role}.crt": cert.fingerprint(hashes.SHA256()).hex()
                                for role, cert in certificates.items()},
        "trust_scope": "application-provided CA only; no system trust registration",
    }
    _write_new(directory / "manifest.json", (json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    return PkiPaths(directory, directory / "ca.crt", directory / "ca.key",
                    directory / "server.crt", directory / "server.key",
                    directory / "client.crt", directory / "client.key", directory / "manifest.json")


def create_test_pki(directory: str | Path, server_names=("localhost", "127.0.0.1"),
                    client_id="test-client") -> PkiPaths:
    """Fresh random test identities, same permission protections, one-day validity."""
    return create_local_pki(directory, server_names=server_names, client_id=client_id, valid_days=1)


def _read_pem(path: str | Path, *, private=False) -> bytes:
    path = Path(path).expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise ValueError("TLS material must be an existing regular file, not a symlink")
    if private:
        _check_private_permissions(path)
    with path.open("rb") as source:
        data = source.read(MAX_PEM_BYTES + 1)
    if not data or len(data) > MAX_PEM_BYTES:
        raise ValueError("TLS PEM files must contain 1 to 65536 bytes")
    return data


def load_tls_material(certificate_path: str | Path, private_key_path: str | Path,
                      ca_path: str | Path, *, role: Literal["server", "client"]) -> TlsMaterial:
    """Preflight one local leaf/key and one direct root; gRPC still verifies peers.

    This helper validates local credentials, not remote hostnames or authorization.
    Pass returned bytes to gRPC mTLS credentials with certificate verification ON.
    """
    if role not in ("server", "client"):
        raise ValueError("TLS role must be server or client")
    x509, hashes, serialization, ec, eku, oid = _crypto()
    from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
    cert_bytes = _read_pem(certificate_path)
    key_bytes = _read_pem(private_key_path, private=True)
    root_bytes = _read_pem(ca_path)
    try:
        certs = x509.load_pem_x509_certificates(cert_bytes)
        roots = x509.load_pem_x509_certificates(root_bytes)
        if len(certs) != 1 or len(roots) != 1:
            raise ValueError("one leaf certificate and one direct root CA are supported")
        cert, root = certs[0], roots[0]
        key = serialization.load_pem_private_key(key_bytes, password=None)
        public_format = (serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
        if key.public_key().public_bytes(*public_format) != cert.public_key().public_bytes(*public_format):
            raise ValueError("TLS private key does not match its certificate")
        now = datetime.now(timezone.utc)
        for item in (cert, root):
            if not item.not_valid_before_utc <= now <= item.not_valid_after_utc:
                raise ValueError("TLS certificate is expired or not yet valid")
        if not root.extensions.get_extension_for_class(x509.BasicConstraints).value.ca:
            raise ValueError("trusted certificate is not a CA")
        if not root.extensions.get_extension_for_class(x509.KeyUsage).value.key_cert_sign:
            raise ValueError("root certificate cannot sign certificates")
        if cert.extensions.get_extension_for_class(x509.BasicConstraints).value.ca:
            raise ValueError("TLS endpoint certificate must not be a CA")
        if not cert.extensions.get_extension_for_class(x509.KeyUsage).value.digital_signature:
            raise ValueError("TLS endpoint certificate cannot sign")
        intended = eku.SERVER_AUTH if role == "server" else eku.CLIENT_AUTH
        if intended not in cert.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value:
            raise ValueError("TLS certificate has the wrong extended key usage")
        cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
        root.verify_directly_issued_by(root)
        cert.verify_directly_issued_by(root)
    except (ValueError, TypeError, x509.ExtensionNotFound, InvalidSignature, UnsupportedAlgorithm) as exc:
        raise ValueError(f"invalid local {role} TLS material: {exc}") from exc
    return TlsMaterial(cert_bytes, key_bytes, root_bytes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--server-name", action="append", required=True,
                        help="server DNS name or bare IP; repeat for additional names")
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--valid-days", type=int, default=30)
    args = parser.parse_args()
    try:
        paths = create_local_pki(args.output_dir, server_names=tuple(args.server_name),
                                 client_id=args.client_id, valid_days=args.valid_days)
    except Exception as exc:
        parser.exit(1, f"TLS provisioning failed ({type(exc).__name__}): {exc}\n")
    print(json.dumps({"output_dir": str(paths.directory), "manifest": str(paths.manifest)}))


if __name__ == "__main__":
    main()
