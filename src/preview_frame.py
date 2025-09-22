import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def iron():
    """Return the IRON colormap (256×3) as a matplotlib ListedColormap.

        Parameters:
            None

        Raises:
            None.

        Returns:
            matplotlib.colors.ListedColormap: 
                A 256-color thermography palette ranging from black → deep blue → orange → white,
                suitable for thermal imagery and general quicklooks.
        """
    data = np.array([
        [0.00,0.00,0.00],[0.00,0.00,0.14],[0.00,0.00,0.20],[0.00,0.00,0.26],
        [0.00,0.00,0.32],[0.01,0.00,0.35],[0.02,0.00,0.39],[0.03,0.00,0.42],
        [0.04,0.00,0.45],[0.05,0.00,0.47],[0.08,0.00,0.48],[0.11,0.00,0.50],
        [0.13,0.00,0.52],[0.16,0.00,0.54],[0.19,0.00,0.55],[0.22,0.00,0.56],
        [0.24,0.00,0.57],[0.26,0.00,0.58],[0.28,0.00,0.59],[0.31,0.00,0.59],
        [0.33,0.00,0.60],[0.36,0.00,0.60],[0.38,0.00,0.61],[0.41,0.00,0.61],
        [0.43,0.00,0.61],[0.45,0.00,0.62],[0.48,0.00,0.62],[0.50,0.00,0.62],
        [0.53,0.00,0.62],[0.55,0.00,0.62],[0.57,0.00,0.61],[0.60,0.00,0.61],
        [0.62,0.00,0.61],[0.64,0.00,0.61],[0.65,0.00,0.60],[0.67,0.00,0.60],
        [0.69,0.00,0.60],[0.70,0.00,0.59],[0.71,0.01,0.58],[0.73,0.02,0.58],
        [0.74,0.02,0.58],[0.75,0.02,0.57],[0.76,0.03,0.56],[0.76,0.04,0.56],
        [0.78,0.05,0.55],[0.79,0.07,0.53],[0.80,0.08,0.52],[0.81,0.09,0.50],
        [0.82,0.10,0.47],[0.82,0.11,0.45],[0.83,0.13,0.44],[0.84,0.15,0.40],
        [0.85,0.16,0.38],[0.86,0.18,0.35],[0.87,0.19,0.31],[0.87,0.21,0.26],
        [0.88,0.22,0.21],[0.89,0.24,0.16],[0.89,0.25,0.12],[0.90,0.27,0.10],
        [0.91,0.28,0.08],[0.91,0.30,0.06],[0.92,0.31,0.05],[0.92,0.32,0.04],
        [0.93,0.34,0.03],[0.93,0.35,0.03],[0.93,0.36,0.02],[0.94,0.38,0.02],
        [0.94,0.39,0.01],[0.95,0.40,0.01],[0.95,0.42,0.01],[0.95,0.43,0.00],
        [0.95,0.44,0.00],[0.96,0.45,0.00],[0.96,0.47,0.00],[0.96,0.49,0.00],
        [0.96,0.51,0.00],[0.97,0.52,0.00],[0.97,0.53,0.00],[0.97,0.55,0.00],
        [0.98,0.56,0.00],[0.98,0.57,0.00],[0.98,0.58,0.00],[0.98,0.60,0.00],
        [0.99,0.62,0.00],[0.99,0.64,0.00],[0.99,0.66,0.00],[0.99,0.67,0.00],
        [1.00,0.69,0.00],[1.00,0.70,0.00],[1.00,0.72,0.00],[1.00,0.73,0.00],
        [1.00,0.75,0.00],[1.00,0.76,0.00],[1.00,0.78,0.00],[1.00,0.79,0.00],
        [1.00,0.80,0.01],[1.00,0.82,0.02],[1.00,0.83,0.04],[1.00,0.85,0.05],
        [1.00,0.86,0.06],[1.00,0.87,0.09],[1.00,0.88,0.13],[1.00,0.89,0.15],
        [1.00,0.90,0.20],[1.00,0.91,0.25],[1.00,0.92,0.29],[1.00,0.93,0.35],
        [1.00,0.94,0.40],[1.00,0.95,0.45],[1.00,0.95,0.53],[1.00,0.96,0.58],
        [1.00,0.96,0.64],[1.00,0.97,0.70],[1.00,0.97,0.75],[1.00,0.98,0.80],
        [1.00,0.98,0.85],[1.00,0.99,0.89],[1.00,1.00,0.94],[1.00,1.00,0.98],
    ])
    return ListedColormap(data, name="iron")

def preview_frame(frame, *, title="", native=False, scale=1, clim=None, mask=None, plot_histogram=True,
                  plot_colorbar=True, fig=None, cmap=None, ctitle="", save_file="", dpi=100):

    """Display a 2-D array as an image with optional histogram and colorbar.

        Parameters:
            frame (array-like, shape (H, W)):        Image to display. 
            title (str, optional):                   Figure title (default: "").
            native (bool, optional):                 If True, keep the input dtype; if False, cast to float (default: False).
            scale (float, optional):                 Isotropic or (h, v) pixel scaling for the on-screen layout (default: 1).
            clim (tuple[float, float]):              Display intensity limits (vmin, vmax). If None, auto-compute as mean ± 3σ
            mask (array-like of bool):               Boolean mask used only for autoscaling when `clim` is None (default: None).
            plot_histogram (bool, optional):         If True, add a side histogram panel (default: True).
            plot_colorbar (bool, optional):          If True, add a colorbar panel (default: True).
            fig (None, optional):                    Existing figure to reuse; if None, a new figure is created (default: None).
            cmap (None, optional):                   Colormap to use. If None, uses the built-in IRON colormap.
            ctitle (str, optional):                  Colorbar label text (default: "").
            save_file (str, optional):               If provided, save the figure to this path (300 dpi, tight bbox)
            dpi (int, optional):                     DPI used to convert pixel-based layout to figure inches (default: 100).

        Raises:
            ValueError:
                If `cmap` is provided as an array that is not shape (M, 3).
                
            OSError:
                If saving to `save_file` fails.

        Returns:
            tuple:
                fig (matplotlib.figure.Figure):         The figure handle.
                ax_img (matplotlib.axes.Axes):          The image axes.
                ax_hist (matplotlib.axes.Axes):         The histogram axes, or None if `plot_histogram=False`.
                cbar (matplotlib.colorbar.Colorbar):    The colorbar handle, or None if `plot_colorbar=False`.
        """

    # --- dtype handling ---
    if not native:
        frame = np.asarray(frame, dtype=float)
    else:
        frame = np.asarray(frame)

    # --- scaling range (clim) ---
    user_range = clim is not None
    if not user_range:
        # autoscale from (masked) finite values: mean ± 3σ
        if mask is not None:
            mask_bool = np.asarray(mask, dtype=bool)
            reduced = frame[mask_bool]
        else:
            reduced = frame.ravel()
        reduced = reduced[np.isfinite(reduced)]
        if reduced.size:
            m = reduced.mean()
            s = reduced.std()
            clim = (m - 3.0 * s, m + 3.0 * s)
        else:
            clim = (np.nanmin(frame), np.nanmax(frame))
    vmin, vmax = float(clim[0]), float(clim[1])

    # --- figure layout in *pixels* (like MATLAB) ---
    H, W = frame.shape[:2]
    if np.ndim(scale) == 0:
        hScale = vScale = float(scale)
    else:
        hScale = float(scale[0])
        vScale = float(scale[1])
    plot_w_px = int(round(W * hScale))
    plot_h_px = int(round(H * vScale))

    hist_w_px = 60 if plot_histogram else 0
    cbar_w_px = 25 if plot_colorbar else 0
    margin_px = 20

    total_w_px = plot_w_px + hist_w_px + cbar_w_px + 3 * margin_px
    total_h_px = plot_h_px + 2 * margin_px

    # inches for matplotlib
    fig_w_in = total_w_px / dpi
    fig_h_in = total_h_px / dpi

    # figure create/reuse
    close_after = False
    if save_file:
        plt.ioff()
        fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi, facecolor="w")
        close_after = True
    elif fig is None:
        fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi, facecolor="w")
    else:
        # reuse passed-in figure
        fig.clf()
        fig.set_size_inches(fig_w_in, fig_h_in, forward=True)
        fig.set_facecolor("w")

    # helper to convert pixels → normalized [0..1] rect
    def px_rect(x, y, w, h):
        return [x / total_w_px, y / total_h_px, w / total_w_px, h / total_h_px]

    # --- image axes ---
    ax_img = fig.add_axes(px_rect(margin_px, margin_px, plot_w_px, plot_h_px))
    ax_img.set_xlim(0.5, W + 0.5)
    ax_img.set_ylim(0.5, H + 0.5)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_ylim(ax_img.get_ylim()[::-1])  # YDir='reverse'
    ax_img.set_title(title, fontsize=12 if title else 0)

    # colormap
    if cmap is None:
        cm = iron()
    else:
        # allow Mx3 arrays
        if isinstance(cmap, (list, tuple, np.ndarray)):
            cmap = np.asarray(cmap)
            if cmap.ndim == 2 and cmap.shape[1] == 3:
                cm = ListedColormap(cmap)
            else:
                raise ValueError("cmap array must be Mx3")
        else:
            cm = cmap

    im = ax_img.imshow(
        frame, cmap=cm, interpolation="nearest",
        origin="upper", vmin=vmin, vmax=vmax, aspect="auto"
    )

    # --- histogram axes ---
    ax_hist = None
    if plot_histogram:
        ax_hist = fig.add_axes(px_rect(2 * margin_px + plot_w_px, margin_px, hist_w_px, plot_h_px))
        edges = np.linspace(vmin, vmax, 128)
        counts, edges = np.histogram(frame.ravel(), bins=edges)
        counts = counts.astype(float)
        if counts.sum() > 0:
            counts /= counts.sum()
        centers = edges[:-1] + 0.5 * (edges[1] - edges[0])
        ax_hist.plot(counts, centers, "k", linewidth=1)
        ax_hist.set_ylim(vmin, vmax)
        ax_hist.set_xlim(left=0)
        ax_hist.set_xticks([])
        ax_hist.yaxis.tick_right()
        ax_hist.yaxis.set_label_position("right")

    # --- colorbar ---
    cbar = None
    if plot_colorbar:
        cbar_ax = fig.add_axes(px_rect(2 * margin_px + plot_w_px + hist_w_px, margin_px, cbar_w_px, plot_h_px))
        cbar = fig.colorbar(im, cax=cbar_ax)
        if ctitle:
            cbar.set_label(ctitle, fontsize=12)

    # --- save ---
    if save_file:
        fig.savefig(save_file, dpi=300, bbox_inches="tight")
        if close_after:
            plt.close(fig)

    # return handles for further tweaking
    return fig, ax_img, ax_hist, cbar
