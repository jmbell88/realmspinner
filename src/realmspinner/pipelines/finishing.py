"""Advisory comparison of source and finished meshes in shared camera frames."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def compare_geometry(source: Path, finished: Path, resolution: int = 256) -> dict[str, Any]:
    try:
        import numpy as np

        from .. import meshaudit

        source_vertices, source_faces = meshaudit.load_mesh(source)
        vertices, faces = meshaudit.load_mesh(finished)
        positions = np.concatenate((source_vertices, vertices))
        if not len(source_faces) or not len(faces) or not np.isfinite(positions).all():
            raise ValueError("empty or nonfinite mesh")
        views = []
        for direction in meshaudit.DEFAULT_VIEWS:
            before = meshaudit._coverage(positions, source_faces, direction, resolution)
            after = meshaudit._coverage(
                positions, faces + len(source_vertices), direction, resolution
            )
            union = int((before | after).sum())
            if not union or not before.any() or not after.any():
                raise ValueError("empty silhouette")
            openings, _ = meshaudit._enclosed_gaps(before)
            opening_pixels = int(openings.sum())
            views.append(
                {
                    "direction": list(direction),
                    "iou": float((before & after).sum() / union),
                    "lost_coverage": float((before & ~after).sum() / before.sum()),
                    "added_coverage": float((after & ~before).sum() / before.sum()),
                    "closed_openings": (
                        float((openings & after).sum() / opening_pixels) if opening_pixels else None
                    ),
                }
            )
        return {
            "measured": True,
            "resolution": resolution,
            "views": views,
            "worst_lost_coverage": max(v["lost_coverage"] for v in views),
            "worst_added_coverage": max(v["added_coverage"] for v in views),
            "source_triangles": len(source_faces),
            "triangles": len(faces),
        }
    except Exception as exc:
        return {"measured": False, "error": str(exc)}
