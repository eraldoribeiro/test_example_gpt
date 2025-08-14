import math
from typing import Optional, Tuple, Literal, Dict, Any

import numpy as np
import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    MeshRenderer, MeshRasterizer, SoftPhongShader, RasterizationSettings,
    PointLights, FoVPerspectiveCameras, PerspectiveCameras
)
from pytorch3d.renderer.cameras import look_at_view_transform, CamerasBase


@torch.no_grad()
def _fibonacci_viewsphere(
    n: int,
    dist: float,
    device: torch.device,
    *,
    hemisphere: Literal["full", "upper", "lower"] = "full",
    min_polar_deg: float = 1.0,
    add_equator: bool = True,
    equator_k: int = 16,
    up: Tuple[float, float, float] = (0.0, 1.0, 0.0),
):
    """
    Near-uniform directions on a sphere via a Fibonacci spiral + optional equator ring.
    Returns batched (R, T) and (elev_deg, azim_deg). M may be >= n if equator added.
    """
    i = torch.arange(n, device=device, dtype=torch.float32)
    ga = math.pi * (3.0 - math.sqrt(5.0))  # golden angle
    y = 1.0 - 2.0 * (i + 0.5) / float(n)   # (-1,1) avoids exact poles/equator
    r = torch.clamp(1.0 - y * y, min=0.0).sqrt()
    theta = ga * i
    x = r * torch.cos(theta)
    z = r * torch.sin(theta)

    elev = torch.asin(y) * (180.0 / math.pi)         # [-90, +90]
    azim = torch.atan2(z, x) * (180.0 / math.pi)     # (-180, 180]

    # Hemisphere filter
    if hemisphere == "upper":
        keep = elev >= 0.0
    elif hemisphere == "lower":
        keep = elev <= 0.0
    else:
        keep = torch.ones_like(elev, dtype=torch.bool)

    # Trim tiny caps near poles for numerical stability
    if min_polar_deg > 0:
        keep = keep & (elev.abs() <= (90.0 - min_polar_deg))

    elev = elev[keep]
    azim = azim[keep]

    # Optional: guaranteed horizontal views
    if add_equator and equator_k > 0:
        az_eq = torch.linspace(-180.0, 180.0, steps=equator_k + 1, device=device)[:-1]
        el_eq = torch.zeros_like(az_eq)
        elev = torch.cat([elev, el_eq], dim=0)
        azim = torch.cat([azim, az_eq], dim=0)

    R, T = look_at_view_transform(
        dist=dist, elev=elev, azim=azim, device=device,
        up=torch.tensor(up, dtype=torch.float32, device=device)[None, :]
    )
    return R, T, elev, azim


@torch.no_grad()
def render_templates_from_obj(
    obj_path: str,
    *,
    device: Optional[torch.device] = None,
    image_size: Tuple[int, int] = (480, 640),           # (H, W)
    lights: Optional[PointLights] = None,
    renderer: Optional[MeshRenderer] = None,            # if you have a custom renderer, pass it
    cameras: Optional[CamerasBase] = None,              # use your prebuilt camera(s) as-is
    # If cameras is None, we will generate views using the options below:
    mode: Literal["fibonacci", "turntable"] = "fibonacci",
    num_views: int = 64,
    dist: float = 3.0,
    fov_degrees: float = 60.0,                          # used only if we create FoV cameras
    hemisphere: Literal["full", "upper", "lower"] = "full",
    min_polar_deg: float = 1.0,
    add_equator: bool = True,
    equator_k: int = 16,
    turntable_elev_deg: float = 30.0,
    center_and_scale: bool = True,                      # normalize mesh to fit a unit box
) -> Dict[str, Any]:
    """
    Render a batch of RGB templates + depth maps from a CAD OBJ using a full viewsphere or turntable.

    You can pass your own `cameras` (any PyTorch3D CamerasBase). If not provided,
    the function generates FoV cameras using either a Fibonacci sphere sampler
    (near-uniform + optional equator ring) or a turntable sweep.

    Returns a dict with:
        - 'rgb':   (B, H, W, 3) float32 in [0,1] on CPU (numpy)
        - 'depth': (B, H, W)   float32 (metric Z_cam) on CPU (torch)
        - 'R':     (B, 3, 3)   torch (device=device)
        - 'T':     (B, 3)      torch (device=device)
        - 'elev_deg': (B,)     torch (device=device) (for generated cameras; zeros if cameras passed)
        - 'azim_deg': (B,)     torch (device=device)
        - 'mesh':  the (possibly normalized) Meshes object on `device`
        - 'cameras': the CamerasBase used
    """
    # ---------- Devices & sizes ----------
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    H, W = int(image_size[0]), int(image_size[1])

    # ---------- Load mesh (.obj with textures if present) ----------
    mesh: Meshes = load_objs_as_meshes([obj_path], device=device)
    if center_and_scale:
        # Normalize to unit cube centered at origin (keeps units consistent for dist)
        verts = mesh.verts_packed()
        vmin, vmax = verts.min(0).values, verts.max(0).values
        center = (vmin + vmax) / 2.0
        scale = (vmax - vmin).max()
        verts_norm = (verts - center) / (scale + 1e-8)
        mesh = Meshes(verts=[verts_norm], faces=mesh.faces_list(), textures=mesh.textures)

    # ---------- Build renderer (if not provided) ----------
    if renderer is None:
        raster_settings = RasterizationSettings(
            image_size=(H, W),
            blur_radius=0.0,
            faces_per_pixel=1
        )
        # If no lights supplied, make a simple point light
        if lights is None:
            lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
        # Create a placeholder camera if we also don't have `cameras` yet; updated below
        _cams_placeholder = FoVPerspectiveCameras(device=device, fov=fov_degrees)
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=_cams_placeholder, raster_settings=raster_settings),
            shader=SoftPhongShader(device=device, cameras=_cams_placeholder, lights=lights),
        )
    else:
        # If a renderer is passed, nudge its lights if possible (optional)
        if (lights is not None) and hasattr(lights, "location"):
            lights.location = torch.tensor([[0.0, 0.0, -3.0]], device=device)

    # ---------- Camera setup ----------
    if cameras is not None:
        cams = cameras
        # Infer batch size from R if available; else assume num_views
        try:
            B = cams.R.shape[0]
        except Exception:
            B = num_views
        # Dummy angles for bookkeeping
        elev_deg = torch.zeros((B,), dtype=torch.float32, device=device)
        azim_deg = torch.zeros((B,), dtype=torch.float32, device=device)
        R_use = cams.R if hasattr(cams, "R") else None
        T_use = cams.T if hasattr(cams, "T") else None
    else:
        if mode == "turntable":
            elev = torch.full((num_views,), float(turntable_elev_deg), device=device, dtype=torch.float32)
            azim = torch.linspace(-180.0, 180.0, steps=num_views, device=device, dtype=torch.float32)
            R_use, T_use = look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
            elev_deg, azim_deg = elev, azim
        elif mode == "fibonacci":
            # Generate potentially more than needed (equator ring), then trim to num_views
            R_try, T_try, elev_try, azim_try = _fibonacci_viewsphere(
                n=num_views, dist=dist, device=device,
                hemisphere=hemisphere, min_polar_deg=min_polar_deg,
                add_equator=add_equator, equator_k=equator_k
            )
            # Trim / pad to exactly num_views for consistency
            M = R_try.shape[0]
            if M >= num_views:
                R_use, T_use = R_try[:num_views], T_try[:num_views]
                elev_deg, azim_deg = elev_try[:num_views], azim_try[:num_views]
            else:
                reps = num_views - M
                R_use = torch.cat([R_try, R_try[:reps]], dim=0)
                T_use = torch.cat([T_try, T_try[:reps]], dim=0)
                elev_deg = torch.cat([elev_try, elev_try[:reps]], dim=0)
                azim_deg = torch.cat([azim_try, azim_try[:reps]], dim=0)
        else:
            raise ValueError("mode must be 'fibonacci' or 'turntable'")

        # Build FoV cameras for our generated poses
        cams = FoVPerspectiveCameras(device=device, R=R_use, T=T_use, fov=fov_degrees)

    B = getattr(cams.R, "shape", [num_views])[0] if hasattr(cams, "R") else num_views

    # ---------- Batch mesh and render ----------
    meshes = mesh.extend(B)

    # Make sure the renderer’s internal cameras are overridden by ours
    images = renderer(meshes, cameras=cams, lights=lights)        # (B, H, W, 4)
    rgb = images[..., :3].detach().cpu().numpy()                  # (B, H, W, 3) in [0,1]

    # Depth via rasterizer (uses same cameras / rasterization settings)
    fragments = renderer.rasterizer(meshes)
    depth = fragments.zbuf[..., 0]                                # (B, H, W), 0 where background
    depth = depth.detach().cpu()

    return {
        "rgb": rgb,                    # np float32 (B,H,W,3) in [0,1]
        "depth": depth,                # torch float32 (B,H,W) metric Z_cam (0 = background)
        "R": getattr(cams, "R", None), # torch (B,3,3) if available
        "T": getattr(cams, "T", None), # torch (B,3)   if available
        "elev_deg": locals().get("elev_deg", torch.zeros((B,), device=device)),
        "azim_deg": locals().get("azim_deg", torch.zeros((B,), device=device)),
        "mesh": mesh,
        "cameras": cams,
    }





# Suppose you already built `my_cameras` (PerspectiveCameras or FoVPerspectiveCameras)
out = render_templates_from_obj(
    "assets/sse_only_real_texture2.obj",
    cameras=cameras,                 # used as-is
    renderer=None,                      # or pass your custom MeshRenderer
    device=torch.device("cuda:0"),
    image_size=(480, 640)
)
rgb_batch = out["rgb"]     # (B,H,W,3)
depth_batch = out["depth"] # (B,H,W)








