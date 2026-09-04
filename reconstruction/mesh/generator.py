"""
Mesh Generator — Converts dense point clouds into textured 3D meshes.
Supports Poisson surface reconstruction and Ball Pivoting.
Exports to OBJ, PLY, and GLB formats.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
from loguru import logger


class MeshGenerator:
    """Generate 3D meshes from point clouds."""

    def __init__(self, method: str = "poisson"):
        """
        Args:
            method: 'poisson' or 'ball_pivoting'
        """
        self.method = method

    def generate(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        normals: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Generate a mesh from a point cloud.

        Returns:
            dict with 'vertices', 'faces', 'vertex_colors' as numpy arrays,
            and 'mesh' as the Open3D mesh object (if available).
        """
        try:
            import open3d as o3d
            return self._generate_open3d(points, colors, normals)
        except ImportError:
            logger.warning("Open3D not available. Using trimesh fallback.")
            return self._generate_trimesh(points, colors)

    def _generate_open3d(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        normals: Optional[np.ndarray] = None,
    ) -> dict:
        """Generate mesh using Open3D."""
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))

        # Estimate normals if not provided
        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals)
        else:
            logger.info("Estimating point normals...")
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            pcd.orient_normals_consistent_tangent_plane(k=15)

        if self.method == "poisson":
            logger.info("Running Poisson surface reconstruction...")
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=9, width=0, scale=1.1, linear_fit=False
            )

            # Remove low-density vertices (cleaning)
            densities = np.asarray(densities)
            threshold = np.quantile(densities, 0.05)
            vertices_to_remove = densities < threshold
            mesh.remove_vertices_by_mask(vertices_to_remove)

        elif self.method == "ball_pivoting":
            logger.info("Running Ball Pivoting reconstruction...")
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radii = [avg_dist * f for f in [0.5, 1.0, 2.0, 4.0]]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii)
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Transfer colors from point cloud to mesh vertices
        mesh.compute_vertex_normals()

        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        vertex_colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else colors[:len(vertices)]

        logger.info(
            f"Mesh generated: {len(vertices):,} vertices, {len(faces):,} faces"
        )

        return {
            "vertices": vertices,
            "faces": faces,
            "vertex_colors": vertex_colors,
            "mesh": mesh,
            "vertex_count": len(vertices),
            "face_count": len(faces),
        }

    def _generate_trimesh(
        self,
        points: np.ndarray,
        colors: np.ndarray,
    ) -> dict:
        """Fallback mesh generation using trimesh (Delaunay-based)."""
        try:
            import trimesh
            from scipy.spatial import Delaunay

            logger.info("Generating mesh via Delaunay triangulation (trimesh)...")

            # Project to 2D for Delaunay (use XY plane)
            if len(points) < 4:
                return {"vertices": points, "faces": np.array([]), "vertex_colors": colors,
                        "mesh": None, "vertex_count": len(points), "face_count": 0}

            tri = Delaunay(points[:, :2])
            faces = tri.simplices

            mesh = trimesh.Trimesh(
                vertices=points,
                faces=faces,
                vertex_colors=(np.clip(colors, 0, 1) * 255).astype(np.uint8),
            )

            logger.info(
                f"Trimesh generated: {len(points):,} vertices, {len(faces):,} faces"
            )

            return {
                "vertices": points,
                "faces": faces,
                "vertex_colors": colors,
                "mesh": mesh,
                "vertex_count": len(points),
                "face_count": len(faces),
            }
        except ImportError:
            logger.error("Neither Open3D nor trimesh available!")
            return {"vertices": points, "faces": np.array([]), "vertex_colors": colors,
                    "mesh": None, "vertex_count": len(points), "face_count": 0}

    def save_mesh(
        self,
        mesh_data: dict,
        output_dir: Path,
        base_name: str = "model",
    ) -> dict:
        """Save mesh in multiple formats."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = {}

        mesh = mesh_data.get("mesh")

        if mesh is not None:
            try:
                import open3d as o3d
                if isinstance(mesh, o3d.geometry.TriangleMesh):
                    # PLY
                    ply_path = output_dir / f"{base_name}.ply"
                    o3d.io.write_triangle_mesh(str(ply_path), mesh)
                    saved_files["ply"] = str(ply_path)

                    # OBJ
                    obj_path = output_dir / f"{base_name}.obj"
                    o3d.io.write_triangle_mesh(str(obj_path), mesh)
                    saved_files["obj"] = str(obj_path)

                    logger.info(f"Saved mesh: PLY and OBJ to {output_dir}")
            except (ImportError, Exception) as e:
                logger.warning(f"Open3D save failed: {e}")

            try:
                import trimesh as tm
                if isinstance(mesh, tm.Trimesh):
                    ply_path = output_dir / f"{base_name}.ply"
                    mesh.export(str(ply_path))
                    saved_files["ply"] = str(ply_path)

                    glb_path = output_dir / f"{base_name}.glb"
                    mesh.export(str(glb_path))
                    saved_files["glb"] = str(glb_path)

                    logger.info(f"Saved mesh: PLY and GLB to {output_dir}")
            except (ImportError, Exception) as e:
                logger.warning(f"Trimesh save failed: {e}")

        # Fallback: write OBJ manually
        if not saved_files:
            obj_path = output_dir / f"{base_name}.obj"
            self._write_obj_manual(
                mesh_data["vertices"],
                mesh_data.get("faces", np.array([])),
                obj_path,
            )
            saved_files["obj"] = str(obj_path)

        return saved_files

    def _write_obj_manual(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        output_path: Path,
    ):
        """Write a minimal OBJ file."""
        with open(output_path, "w") as f:
            f.write("# Generated by Drone 3D Reconstruction System\n")
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in faces:
                # OBJ is 1-indexed
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        logger.info(f"Saved OBJ (manual): {output_path}")
