"""Optional OCI SDK transfer helpers; resource principals work inside OCI."""

from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit


def parse_uri(uri):
    parsed = urlsplit(uri)
    if parsed.scheme != "oci" or parsed.query or parsed.fragment or parsed.netloc.count("@") != 1:
        raise ValueError("expected oci://bucket@namespace/object-or-prefix")
    bucket, namespace = parsed.netloc.split("@")
    key = parsed.path.lstrip("/")
    if not bucket or not namespace or not key or ".." in PurePosixPath(key).parts:
        raise ValueError("bucket, namespace and safe object path are required")
    return bucket, namespace, key


def client(auth="resource_principal", profile="DEFAULT"):
    import oci
    if auth == "resource_principal":
        signer = oci.auth.signers.get_resource_principals_signer()
        return oci.object_storage.ObjectStorageClient({}, signer=signer)
    if auth == "instance_principal":
        signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
        return oci.object_storage.ObjectStorageClient({}, signer=signer)
    if auth == "config":
        return oci.object_storage.ObjectStorageClient(oci.config.from_file(profile_name=profile))
    raise ValueError("unknown OCI authentication mode")


def upload(path, uri, auth="resource_principal", profile="DEFAULT"):
    import oci
    path = Path(path)
    bucket, namespace, key = parse_uri(uri)
    storage = client(auth, profile)
    # The upload manager selects multipart uploads for multi-GB artifacts.
    manager = oci.object_storage.UploadManager(storage, allow_parallel_uploads=True, parallel_process_count=4)
    if path.is_file():
        files = [(path, key)]
    elif path.is_dir():
        files = [(file, key.rstrip("/")+"/"+file.relative_to(path).as_posix())
                 for file in sorted(path.rglob("*")) if file.is_file()]
    else:
        raise FileNotFoundError(path)
    for file, object_name in files:
        manager.upload_file(namespace, bucket, object_name, str(file))
    return {"files": len(files), "uri": uri}


def download(uri, output, auth="resource_principal", profile="DEFAULT"):
    import oci
    bucket, namespace, prefix = parse_uri(uri)
    if not prefix.endswith("/"):
        raise ValueError("download expects a directory prefix ending with /; use OCI CLI for one ZIP")
    root = Path(output)
    if root.exists():
        raise FileExistsError(root)
    storage = client(auth, profile)
    objects = oci.pagination.list_call_get_all_results(storage.list_objects, namespace, bucket, prefix=prefix).data.objects
    if not objects:
        raise ValueError("Object Storage prefix contains no files")
    root.mkdir(parents=True)
    for obj in objects:
        relative = obj.name[len(prefix):]
        if not relative or obj.name.endswith("/"):
            continue
        destination = (root/relative).resolve()
        if not destination.is_relative_to(root.resolve()):
            raise ValueError("unsafe object name")
        destination.parent.mkdir(parents=True, exist_ok=True)
        response = storage.get_object(namespace, bucket, obj.name)
        temporary = destination.with_suffix(destination.suffix+".tmp")
        with temporary.open("wb") as handle:
            for chunk in response.data.raw.stream(8*1024*1024, decode_content=False):
                handle.write(chunk)
        temporary.rename(destination)
    return {"files": len(objects), "output": str(root)}
