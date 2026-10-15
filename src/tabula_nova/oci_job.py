"""Create the archive layout OCI Data Science jobs expect."""

from pathlib import Path
import zipfile


def pack_job(output):
    root = Path(__file__).resolve().parents[2]
    if not (root/"oci"/"train.sh").is_file():
        raise ValueError("pack-job must run from an editable source checkout")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if output.name != "tabula_nova_8b.zip":
        raise ValueError("OCI job archive name must be tabula_nova_8b.zip to match its root folder")
    output.parent.mkdir(parents=True, exist_ok=True)
    candidates = [root/"pyproject.toml", root/"README.md", root/"data/sources.json", root/"data/README.md"]
    for name in ("PROVENANCE.md", "VALIDATION.md", "PROJECT_STATUS.md"):
        if (root/name).is_file():
            candidates.append(root/name)
    for directory in ("src", "configs", "oci", "tools", "docs"):
        candidates.extend(path for path in (root/directory).rglob("*")
                          if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
                          and not any(part.endswith(".egg-info") for part in path.parts))
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(candidates):
            archive.write(path, Path("tabula_nova_8b")/path.relative_to(root))
    return output
